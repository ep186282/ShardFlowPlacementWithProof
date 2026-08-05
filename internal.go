package shardflow

import (
	"fmt"
	"math"
	"sort"
)

type instance struct {
	nodes       []Node
	shards      []Shard
	old         []map[NodeID]struct{}
	nodeIndex   map[NodeID]int
	shardIndex  map[ShardID]int
	domains     []string
	nodeDomains []int
	total       int
}

type arc struct {
	to      int
	reverse int
	cap     int
	initial int
	cost    Cost
}

type network struct {
	adj [][]arc
}

type edgeRef struct {
	from  int
	index int
}

type layout struct {
	source      int
	sink        int
	sourceEdges []edgeRef
	domainEdges []edgeRef
	choiceEdges []edgeRef
	slotEdges   [][]edgeRef
	domains     int
	nodes       int
	vertices    int
}

var invalidEdge = edgeRef{from: -1, index: -1}

// The solver builds an explicit residual network. These limits reject an input
// before its graph can consume unbounded memory or make integer bounds unclear.
const (
	maxReplicas     = 1 << 16
	maxVertices     = 1 << 18
	maxForwardEdges = 1 << 20
)

// Normalize copies and canonicalizes caller-owned input so graph construction
// and output do not depend on input order.
func normalize(nodes []Node, shards []Shard, old Placement) (*instance, error) {
	in := &instance{
		nodes:      append([]Node(nil), nodes...),
		shards:     append([]Shard(nil), shards...),
		nodeIndex:  make(map[NodeID]int, len(nodes)),
		shardIndex: make(map[ShardID]int, len(shards)),
	}
	sort.Slice(in.nodes, func(i, j int) bool { return in.nodes[i].ID < in.nodes[j].ID })
	sort.Slice(in.shards, func(i, j int) bool { return in.shards[i].ID < in.shards[j].ID })

	domainSet := make(map[string]struct{})
	for i, node := range in.nodes {
		if node.ID == "" || node.Domain == "" {
			return nil, fmt.Errorf("node ID and domain must be nonempty")
		}
		if node.Capacity < 0 {
			return nil, fmt.Errorf("node %q has negative capacity", node.ID)
		}
		if i > 0 && in.nodes[i-1].ID == node.ID {
			return nil, fmt.Errorf("duplicate node ID %q", node.ID)
		}
		in.nodeIndex[node.ID] = i
		domainSet[node.Domain] = struct{}{}
	}
	for domain := range domainSet {
		in.domains = append(in.domains, domain)
	}
	sort.Strings(in.domains)
	domainIndex := make(map[string]int, len(in.domains))
	for i, domain := range in.domains {
		domainIndex[domain] = i
	}
	in.nodeDomains = make([]int, len(in.nodes))
	for i, node := range in.nodes {
		in.nodeDomains[i] = domainIndex[node.Domain]
	}

	for i, shard := range in.shards {
		if shard.ID == "" {
			return nil, fmt.Errorf("shard ID must be nonempty")
		}
		if shard.Replicas <= 0 {
			return nil, fmt.Errorf("shard %q has nonpositive replica count", shard.ID)
		}
		if i > 0 && in.shards[i-1].ID == shard.ID {
			return nil, fmt.Errorf("duplicate shard ID %q", shard.ID)
		}
		if shard.Replicas > maxReplicas-in.total {
			return nil, fmt.Errorf("total replica count exceeds %d", maxReplicas)
		}
		in.total += shard.Replicas
		in.shardIndex[shard.ID] = i
	}
	if int64(in.total) > 0 && int64(in.total) > math.MaxInt64/int64(in.total) {
		return nil, fmt.Errorf("maximum concentration overflows int64")
	}

	in.old = make([]map[NodeID]struct{}, len(in.shards))
	for i := range in.old {
		in.old[i] = make(map[NodeID]struct{})
	}
	seenShards := make(map[ShardID]struct{}, len(old))
	for _, assignment := range old {
		shard, ok := in.shardIndex[assignment.Shard]
		if !ok {
			// Removed shards need no replicas and charge nothing.
			continue
		}
		if _, duplicate := seenShards[assignment.Shard]; duplicate {
			return nil, fmt.Errorf("old placement repeats shard %q", assignment.Shard)
		}
		seenShards[assignment.Shard] = struct{}{}
		for _, node := range assignment.Nodes {
			if node == "" {
				return nil, fmt.Errorf("old placement contains an empty node ID")
			}
			if _, duplicate := in.old[shard][node]; duplicate {
				return nil, fmt.Errorf("old placement repeats node %q for shard %q",
					node, assignment.Shard)
			}
			in.old[shard][node] = struct{}{}
		}
	}
	return in, nil
}

// Graph dimensions contain shard-by-domain and shard-by-node products, so
// compute and bound every expansion before allocating the residual network.
func networkSizes(in *instance) (int, int, int, error) {
	shardDomains, err := checkedMul(len(in.shards), len(in.domains))
	if err != nil {
		return 0, 0, 0, err
	}
	vertices, err := checkedAdd(2, len(in.shards))
	if err != nil {
		return 0, 0, 0, err
	}
	vertices, err = checkedAdd(vertices, len(in.nodes))
	if err != nil {
		return 0, 0, 0, err
	}
	vertices, err = checkedAdd(vertices, shardDomains)
	if err != nil {
		return 0, 0, 0, err
	}
	choices, err := checkedMul(len(in.shards), len(in.nodes))
	if err != nil {
		return 0, 0, 0, err
	}
	slots := 0
	for _, node := range in.nodes {
		count := min(node.Capacity, len(in.shards))
		slots, err = checkedAdd(slots, count)
		if err != nil {
			return 0, 0, 0, err
		}
	}
	edges, err := checkedAdd(len(in.shards), shardDomains)
	if err != nil {
		return 0, 0, 0, err
	}
	edges, err = checkedAdd(edges, choices)
	if err != nil {
		return 0, 0, 0, err
	}
	edges, err = checkedAdd(edges, slots)
	if err != nil {
		return 0, 0, 0, err
	}
	if vertices > maxVertices || edges > maxForwardEdges {
		return 0, 0, 0, fmt.Errorf(
			"network requires %d vertices and %d edges; limits are %d and %d",
			vertices, edges, maxVertices, maxForwardEdges,
		)
	}
	return shardDomains, vertices, choices, nil
}

// inspectPlacement validates a complete assignment while reconstructing its
// exact lexicographic objective.
func inspectPlacement(in *instance, placement Placement) ([]bool, []int, Cost, error) {
	if len(placement) != len(in.shards) {
		return nil, nil, Cost{}, fmt.Errorf("placement has %d shards, want %d",
			len(placement), len(in.shards))
	}
	selected, err := makeBools(len(in.shards), len(in.nodes))
	if err != nil {
		return nil, nil, Cost{}, err
	}
	loads := make([]int, len(in.nodes))
	seenShards := make([]bool, len(in.shards))
	objective := Cost{}

	for _, assignment := range placement {
		shard, ok := in.shardIndex[assignment.Shard]
		if !ok {
			return nil, nil, Cost{}, fmt.Errorf("placement references unknown shard %q",
				assignment.Shard)
		}
		if seenShards[shard] {
			return nil, nil, Cost{}, fmt.Errorf("placement repeats shard %q", assignment.Shard)
		}
		seenShards[shard] = true
		if len(assignment.Nodes) != in.shards[shard].Replicas {
			return nil, nil, Cost{}, fmt.Errorf("shard %q has %d replicas, want %d",
				assignment.Shard, len(assignment.Nodes), in.shards[shard].Replicas)
		}
		seenDomains := make([]bool, len(in.domains))
		for _, nodeID := range assignment.Nodes {
			node, ok := in.nodeIndex[nodeID]
			if !ok {
				return nil, nil, Cost{}, fmt.Errorf("shard %q uses unknown node %q",
					assignment.Shard, nodeID)
			}
			offset := shard*len(in.nodes) + node
			if selected[offset] {
				return nil, nil, Cost{}, fmt.Errorf("shard %q repeats node %q",
					assignment.Shard, nodeID)
			}
			domain := in.nodeDomains[node]
			if seenDomains[domain] {
				return nil, nil, Cost{}, fmt.Errorf("shard %q repeats domain %q",
					assignment.Shard, in.domains[domain])
			}
			seenDomains[domain] = true
			selected[offset] = true
			loads[node]++
			if loads[node] > in.nodes[node].Capacity {
				return nil, nil, Cost{}, fmt.Errorf("node %q exceeds capacity",
					in.nodes[node].ID)
			}
			if _, retained := in.old[shard][nodeID]; !retained {
				objective.Moves, err = add64(objective.Moves, 1)
				if err != nil {
					return nil, nil, Cost{}, err
				}
			}
		}
	}
	for shard, seen := range seenShards {
		if !seen {
			return nil, nil, Cost{}, fmt.Errorf("placement omits shard %q", in.shards[shard].ID)
		}
	}
	for _, load := range loads {
		square, err := checkedSquare(int64(load))
		if err != nil {
			return nil, nil, Cost{}, err
		}
		objective.Concentration, err = add64(objective.Concentration, square)
		if err != nil {
			return nil, nil, Cost{}, err
		}
	}
	return selected, loads, objective, nil
}

// applyPlacement reconstructs the flow represented by a placement in a fresh
// residual network.
func applyPlacement(
	graph *network,
	shape *layout,
	in *instance,
	selected []bool,
	loads []int,
) error {
	for shard, item := range in.shards {
		if err := graph.push(shape.sourceEdges[shard], item.Replicas); err != nil {
			return err
		}
		for node := range in.nodes {
			if !selected[shard*shape.nodes+node] {
				continue
			}
			domain := in.nodeDomains[node]
			if err := graph.push(shape.domainEdges[shard*shape.domains+domain], 1); err != nil {
				return err
			}
			if err := graph.push(shape.choiceEdges[shard*shape.nodes+node], 1); err != nil {
				return err
			}
		}
	}
	for node, load := range loads {
		if load > len(shape.slotEdges[node]) {
			return fmt.Errorf("node %q uses unavailable slot", in.nodes[node].ID)
		}
		for slot := 0; slot < load; slot++ {
			if err := graph.push(shape.slotEdges[node][slot], 1); err != nil {
				return err
			}
		}
	}
	return nil
}

// A saturated choice edge records the node selected by one unit of integral
// flow. Reading those edges recovers the corresponding shard placement.
func extractPlacement(graph *network, shape *layout, in *instance) (Placement, error) {
	placement := make(Placement, len(in.shards))
	for shard, item := range in.shards {
		placement[shard].Shard = item.ID
		for node := range in.nodes {
			ref := shape.choiceEdges[shard*shape.nodes+node]
			if ref == invalidEdge {
				continue
			}
			edge := graph.adj[ref.from][ref.index]
			if edge.initial-edge.cap == 1 {
				placement[shard].Nodes = append(placement[shard].Nodes, in.nodes[node].ID)
			}
		}
		sort.Slice(placement[shard].Nodes, func(i, j int) bool {
			return placement[shard].Nodes[i] < placement[shard].Nodes[j]
		})
		if len(placement[shard].Nodes) != item.Replicas {
			return nil, fmt.Errorf("shard %q has %d extracted replicas, want %d",
				item.ID, len(placement[shard].Nodes), item.Replicas)
		}
	}
	return placement, nil
}

// Only original forward edges contribute to a cut. Residual reverse edges have
// zero initial capacity and are excluded.
func cutCapacity(graph *network, inside []bool) (int64, error) {
	var capacity int64
	for from, edges := range graph.adj {
		if !inside[from] {
			continue
		}
		for _, edge := range edges {
			if edge.initial == 0 || inside[edge.to] {
				continue
			}
			var err error
			capacity, err = add64(capacity, int64(edge.initial))
			if err != nil {
				return 0, err
			}
		}
	}
	return capacity, nil
}

func (graph *network) addEdge(from, to, capacity int, cost Cost) (edgeRef, error) {
	reverseCost, err := cost.neg()
	if err != nil {
		return invalidEdge, err
	}
	ref := edgeRef{from: from, index: len(graph.adj[from])}
	forward := arc{
		to:      to,
		reverse: len(graph.adj[to]),
		cap:     capacity,
		initial: capacity,
		cost:    cost,
	}
	reverse := arc{
		to:      from,
		reverse: len(graph.adj[from]),
		cost:    reverseCost,
	}
	graph.adj[from] = append(graph.adj[from], forward)
	graph.adj[to] = append(graph.adj[to], reverse)
	return ref, nil
}

func (graph *network) push(ref edgeRef, amount int) error {
	if ref == invalidEdge {
		return fmt.Errorf("missing network edge")
	}
	edge := &graph.adj[ref.from][ref.index]
	if amount < 0 || edge.cap < amount {
		return fmt.Errorf("edge %d to %d has capacity %d, need %d",
			ref.from, edge.to, edge.cap, amount)
	}
	to, reverse := edge.to, edge.reverse
	edge.cap -= amount
	graph.adj[to][reverse].cap += amount
	return nil
}

func add64(a, b int64) (int64, error) {
	if b > 0 && a > math.MaxInt64-b || b < 0 && a < math.MinInt64-b {
		return 0, fmt.Errorf("integer addition overflows int64")
	}
	return a + b, nil
}

func sub64(a, b int64) (int64, error) {
	if b > 0 && a < math.MinInt64+b || b < 0 && a > math.MaxInt64+b {
		return 0, fmt.Errorf("integer subtraction overflows int64")
	}
	return a - b, nil
}

func checkedSquare(value int64) (int64, error) {
	if value < 0 || value > 0 && value > math.MaxInt64/value {
		return 0, fmt.Errorf("integer multiplication overflows int64")
	}
	return value * value, nil
}

func checkedAdd(a, b int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if a < 0 || b < 0 || a > maxInt-b {
		return 0, fmt.Errorf("graph size overflows int")
	}
	return a + b, nil
}

func checkedMul(a, b int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if a < 0 || b < 0 || a > 0 && b > maxInt/a {
		return 0, fmt.Errorf("graph size overflows int")
	}
	return a * b, nil
}

func makeBools(rows, columns int) ([]bool, error) {
	size, err := checkedMul(rows, columns)
	if err != nil {
		return nil, err
	}
	return make([]bool, size), nil
}

// A zero edgeRef can name the graph's first edge, so absent entries use an
// explicit sentinel.
func filledRefs(size int) []edgeRef {
	refs := make([]edgeRef, size)
	for i := range refs {
		refs[i] = invalidEdge
	}
	return refs
}

type queueItem struct {
	vertex   int
	distance Cost
}

type priorityQueue []queueItem

func (queue priorityQueue) Len() int { return len(queue) }

func (queue priorityQueue) Less(i, j int) bool {
	if queue[i].distance != queue[j].distance {
		return queue[i].distance.Less(queue[j].distance)
	}
	return queue[i].vertex < queue[j].vertex
}

func (queue priorityQueue) Swap(i, j int) {
	queue[i], queue[j] = queue[j], queue[i]
}

func (queue *priorityQueue) Push(value any) {
	*queue = append(*queue, value.(queueItem))
}

func (queue *priorityQueue) Pop() any {
	old := *queue
	last := old[len(old)-1]
	*queue = old[:len(old)-1]
	return last
}
