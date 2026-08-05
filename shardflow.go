// Package shardflow solves cluster rebalancing as min-cost flow. It places
// replicated shards under capacity and failure-domain constraints, minimizing
// data movement before load concentration. Placement is deterministic.
//
// Every result carries a proof. A solution includes vertex potentials making
// residual edge costs nonnegative, certifying optimality. An infeasible result
// includes a source-side cut with capacity below the number of required replicas.
// Verify checks either certificate without rerunning Place.
package shardflow

import (
	"container/heap"
	"fmt"
	"math"
)

// Cost is ordered lexicographically, with moves first, then concentration.
// Storing both terms avoids encoding that order in a weighted scalar.
type Cost struct {
	Moves         int64
	Concentration int64
}

// Less reports whether c precedes other in lexicographic order.
func (c Cost) Less(other Cost) bool {
	return c.Moves < other.Moves ||
		c.Moves == other.Moves && c.Concentration < other.Concentration
}

// Place returns a minimum-cost placement or a cut proving infeasibility.
func Place(nodes []Node, shards []Shard, old Placement) (Result, error) {
	in, err := normalize(nodes, shards, old)
	if err != nil {
		return nil, err
	}
	graph, shape, err := buildNetwork(in)
	if err != nil {
		return nil, err
	}

	flow, reached, err := augment(graph, shape.source, shape.sink, in.total)
	if err != nil {
		return nil, err
	}
	if flow != in.total {
		capacity, err := cutCapacity(graph, reached)
		if err != nil {
			return nil, err
		}
		cut := make([]int, 0)
		for vertex, inside := range reached {
			if inside {
				cut = append(cut, vertex)
			}
		}
		return Infeasible{
			Cut:          cut,
			RequiredFlow: int64(in.total),
			CutCapacity:  capacity,
		}, nil
	}

	placement, err := extractPlacement(graph, shape, in)
	if err != nil {
		return nil, err
	}
	_, _, objective, err := inspectPlacement(in, placement)
	if err != nil {
		return nil, fmt.Errorf("internal placement: %w", err)
	}
	potentials, err := certificatePotentials(graph)
	if err != nil {
		return nil, err
	}
	return Solution{
		Placement:  placement,
		Objective:  objective,
		Potentials: potentials,
	}, nil
}

// Verify checks a result without running the placement optimizer.
func Verify(nodes []Node, shards []Shard, old Placement, result Result) error {
	in, err := normalize(nodes, shards, old)
	if err != nil {
		return err
	}
	graph, shape, err := buildNetwork(in)
	if err != nil {
		return err
	}
	switch outcome := result.(type) {
	case Infeasible:
		return verifyCut(graph, shape, in.total, outcome)
	case Solution:
		return verifySolution(graph, shape, in, outcome)
	default:
		return fmt.Errorf("unknown result type %T", result)
	}
}

func verifySolution(graph *network, shape *layout, in *instance, result Solution) error {
	selected, loads, objective, err := inspectPlacement(in, result.Placement)
	if err != nil {
		return err
	}
	if objective != result.Objective {
		return fmt.Errorf("objective is %+v, want %+v", result.Objective, objective)
	}
	// Reconstruct the claimed flow in the fresh network. Its residual edges
	// depend only on the inputs and claimed placement, not on solver state.
	if err := applyPlacement(graph, shape, in, selected, loads); err != nil {
		return fmt.Errorf("reconstruct flow: %w", err)
	}
	if len(result.Potentials) != shape.vertices {
		return fmt.Errorf("got %d potentials, want %d",
			len(result.Potentials), shape.vertices)
	}

	// Any other complete placement changes this flow along residual cycles.
	// Potentials cancel around each cycle, so nonnegative reduced costs prove
	// that none of those changes can lower the objective.
	for from, edges := range graph.adj {
		for _, edge := range edges {
			if edge.cap == 0 {
				continue
			}
			reduced, err := edge.cost.add(result.Potentials[from])
			if err != nil {
				return err
			}
			reduced, err = reduced.sub(result.Potentials[edge.to])
			if err != nil {
				return err
			}
			if reduced.Less(Cost{}) {
				return fmt.Errorf("negative reduced cost on edge %d to %d", from, edge.to)
			}
		}
	}
	return nil
}

func buildNetwork(in *instance) (*network, *layout, error) {
	shardDomains, vertices, choices, err := networkSizes(in)
	if err != nil {
		return nil, nil, err
	}

	// Vertices occupy contiguous ranges for shards, shard-domain pairs, and
	// nodes. Their positions are stable because normalize sorts every input.
	shape := &layout{
		source:      0,
		sink:        vertices - 1,
		sourceEdges: make([]edgeRef, len(in.shards)),
		domainEdges: filledRefs(shardDomains),
		choiceEdges: filledRefs(choices),
		slotEdges:   make([][]edgeRef, len(in.nodes)),
		domains:     len(in.domains),
		nodes:       len(in.nodes),
		vertices:    vertices,
	}
	graph := &network{adj: make([][]arc, vertices)}
	shardBase := 1
	domainBase := shardBase + len(in.shards)
	nodeBase := domainBase + shardDomains

	for shard, item := range in.shards {
		ref, err := graph.addEdge(shape.source, shardBase+shard, item.Replicas, Cost{})
		if err != nil {
			return nil, nil, err
		}
		shape.sourceEdges[shard] = ref
		for domain := range in.domains {
			vertex := domainBase + shard*len(in.domains) + domain
			ref, err := graph.addEdge(shardBase+shard, vertex, 1, Cost{})
			if err != nil {
				return nil, nil, err
			}
			shape.domainEdges[shard*len(in.domains)+domain] = ref
		}
	}

	// Each unit of flow chooses one shard, one failure domain, one node, and
	// one slot. The shard-to-domain capacity prevents two replicas of the same
	// shard from entering one failure domain.
	for shard := range in.shards {
		for node, item := range in.nodes {
			if item.Capacity == 0 {
				// Omitting this choice models a drain while still allowing the
				// node to appear in the old placement.
				continue
			}
			domainVertex := domainBase +
				shard*len(in.domains) + in.nodeDomains[node]
			move := int64(1)
			if _, retained := in.old[shard][item.ID]; retained {
				move = 0
			}
			ref, err := graph.addEdge(domainVertex, nodeBase+node, 1, Cost{Moves: move})
			if err != nil {
				return nil, nil, err
			}
			shape.choiceEdges[shard*len(in.nodes)+node] = ref
		}
	}

	for node, item := range in.nodes {
		// Each shard reaches a node through one unit-capacity choice edge, so a
		// node holds at most one replica per shard regardless of its capacity.
		slots := min(item.Capacity, len(in.shards))
		shape.slotEdges[node] = make([]edgeRef, 0, slots)
		for slot := 1; slot <= slots; slot++ {
			// The first L odd numbers sum to L*L, so filling the first L
			// slot edges contributes exactly this node's concentration cost.
			cost := Cost{Concentration: 2*int64(slot) - 1}
			ref, err := graph.addEdge(nodeBase+node, shape.sink, 1, cost)
			if err != nil {
				return nil, nil, err
			}
			shape.slotEdges[node] = append(shape.slotEdges[node], ref)
		}
	}
	return graph, shape, nil
}

func augment(graph *network, source, sink, required int) (int, []bool, error) {
	// Dijkstra needs nonnegative edge costs and comparisons that addition
	// preserves. Lexicographic Cost values provide the ordering, while vertex
	// potentials keep residual-edge costs nonnegative.
	// Original forward costs are nonnegative, so zero initial potentials work.
	potential := make([]Cost, len(graph.adj))
	// Every augmenting path contains a unit-capacity edge, so one iteration
	// places exactly one replica.
	for flow := 0; flow < required; flow++ {
		distance := make([]Cost, len(graph.adj))
		reached := make([]bool, len(graph.adj))
		previous := filledRefs(len(graph.adj))
		reached[source] = true
		queue := priorityQueue{{vertex: source}}
		heap.Init(&queue)

		for queue.Len() > 0 {
			item := heap.Pop(&queue).(queueItem)
			if !reached[item.vertex] || item.distance != distance[item.vertex] {
				continue
			}
			for edgeIndex, edge := range graph.adj[item.vertex] {
				if edge.cap == 0 {
					continue
				}
				reduced, err := edge.cost.add(potential[item.vertex])
				if err != nil {
					return 0, nil, err
				}
				reduced, err = reduced.sub(potential[edge.to])
				if err != nil {
					return 0, nil, err
				}
				if reduced.Less(Cost{}) {
					return 0, nil, fmt.Errorf("negative search cost on edge %d to %d",
						item.vertex, edge.to)
				}
				next, err := distance[item.vertex].add(reduced)
				if err != nil {
					return 0, nil, err
				}
				if reached[edge.to] && !next.Less(distance[edge.to]) {
					continue
				}
				reached[edge.to] = true
				distance[edge.to] = next
				previous[edge.to] = edgeRef{from: item.vertex, index: edgeIndex}
				heap.Push(&queue, queueItem{vertex: edge.to, distance: next})
			}
		}
		if !reached[sink] {
			// With no augmenting path, the reached vertices form the source
			// side of a cut that bounds every possible placement.
			return flow, reached, nil
		}
		// No positive-capacity edge leaves the reached set, and augmentation adds
		// capacity only within it. Stale potentials outside the set are never used.
		for vertex, ok := range reached {
			if !ok {
				continue
			}
			next, err := potential[vertex].add(distance[vertex])
			if err != nil {
				return 0, nil, err
			}
			potential[vertex] = next
		}
		for vertex := sink; vertex != source; {
			ref := previous[vertex]
			if ref == invalidEdge {
				return 0, nil, fmt.Errorf("missing predecessor for vertex %d", vertex)
			}
			if err := graph.push(ref, 1); err != nil {
				return 0, nil, err
			}
			vertex = ref.from
		}
	}
	return required, nil, nil
}

func certificatePotentials(graph *network) ([]Cost, error) {
	// The search updates potentials only for vertices reachable from the source.
	// Verify checks every residual edge, including disconnected components, so
	// compute a fresh potential assignment for the entire graph.
	// Initializing every distance to zero acts like a temporary zero-cost source
	// connected to every vertex. Once relaxation converges, no residual edge can
	// lower a destination distance, so every reduced cost is nonnegative.
	distance := make([]Cost, len(graph.adj))
	for pass := 0; pass < len(graph.adj); pass++ {
		changed := false
		for from, edges := range graph.adj {
			for _, edge := range edges {
				if edge.cap == 0 {
					continue
				}
				next, err := distance[from].add(edge.cost)
				if err != nil {
					return nil, err
				}
				if next.Less(distance[edge.to]) {
					distance[edge.to] = next
					changed = true
				}
			}
		}
		if !changed {
			return distance, nil
		}
		if pass == len(graph.adj)-1 {
			return nil, fmt.Errorf("residual graph contains a negative-cost cycle")
		}
	}
	return distance, nil
}

func verifyCut(graph *network, shape *layout, required int, proof Infeasible) error {
	if proof.RequiredFlow != int64(required) {
		return fmt.Errorf("cut claims required flow %d, want %d",
			proof.RequiredFlow, required)
	}
	inside := make([]bool, shape.vertices)
	for _, vertex := range proof.Cut {
		if vertex < 0 || vertex >= shape.vertices {
			return fmt.Errorf("cut contains invalid vertex %d", vertex)
		}
		if inside[vertex] {
			return fmt.Errorf("cut repeats vertex %d", vertex)
		}
		inside[vertex] = true
	}
	if !inside[shape.source] || inside[shape.sink] {
		return fmt.Errorf("cut must contain source and exclude sink")
	}

	// Recompute capacity from input edges instead of trusting the partial flow.
	// Capacity below required flow proves that no complete placement exists.
	capacity, err := cutCapacity(graph, inside)
	if err != nil {
		return err
	}
	if capacity != proof.CutCapacity {
		return fmt.Errorf("cut capacity is %d, want %d", proof.CutCapacity, capacity)
	}
	if capacity >= int64(required) {
		return fmt.Errorf("cut capacity %d does not prove infeasibility", capacity)
	}
	return nil
}

func (a Cost) add(b Cost) (Cost, error) {
	moves, err := add64(a.Moves, b.Moves)
	if err != nil {
		return Cost{}, err
	}
	concentration, err := add64(a.Concentration, b.Concentration)
	if err != nil {
		return Cost{}, err
	}
	return Cost{Moves: moves, Concentration: concentration}, nil
}

func (a Cost) sub(b Cost) (Cost, error) {
	moves, err := sub64(a.Moves, b.Moves)
	if err != nil {
		return Cost{}, err
	}
	concentration, err := sub64(a.Concentration, b.Concentration)
	if err != nil {
		return Cost{}, err
	}
	return Cost{Moves: moves, Concentration: concentration}, nil
}

func (a Cost) neg() (Cost, error) {
	if a.Moves == math.MinInt64 || a.Concentration == math.MinInt64 {
		return Cost{}, fmt.Errorf("cost negation overflows int64")
	}
	return Cost{Moves: -a.Moves, Concentration: -a.Concentration}, nil
}
