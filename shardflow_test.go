package shardflow_test

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	sf "shardflow"
)

// Node c is gone. Keeping x on a would strand y, whose retained replica on d
// means its replacement must use rack r1. The residual path moves x to b.
func TestDrainReroutesThroughResidualEdge(t *testing.T) {
	nodes := []sf.Node{
		{ID: "a", Domain: "r1", Capacity: 1},
		{ID: "b", Domain: "r2", Capacity: 1},
		{ID: "d", Domain: "r2", Capacity: 1},
	}
	shards := []sf.Shard{
		{ID: "x", Replicas: 1},
		{ID: "y", Replicas: 2},
	}
	old := sf.Placement{
		{Shard: "x", Nodes: []sf.NodeID{"c"}},
		{Shard: "y", Nodes: []sf.NodeID{"c", "d"}},
	}

	result, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatal(err)
	}
	if err := sf.Verify(nodes, shards, old, result); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	solution := requireSolution(t, result)
	want := sf.Cost{Moves: 2, Concentration: 3}
	if solution.Objective != want {
		t.Fatalf("objective is %+v, want %+v", solution.Objective, want)
	}
	wantPlacement := sf.Placement{
		{Shard: "x", Nodes: []sf.NodeID{"b"}},
		{Shard: "y", Nodes: []sf.NodeID{"a", "d"}},
	}
	if !reflect.DeepEqual(solution.Placement, wantPlacement) {
		t.Fatalf("placement is %+v, want %+v", solution.Placement, wantPlacement)
	}
}

// Three replicas cannot occupy two failure domains without repeating one. The
// returned cut should expose the same capacity obstruction.
func TestInfeasibleCut(t *testing.T) {
	nodes := []sf.Node{
		{ID: "a", Domain: "east", Capacity: 2},
		{ID: "b", Domain: "west", Capacity: 2},
	}
	shards := []sf.Shard{{ID: "x", Replicas: 3}}

	result, err := sf.Place(nodes, shards, nil)
	if err != nil {
		t.Fatal(err)
	}
	proof := requireInfeasible(t, result)
	if err := sf.Verify(nodes, shards, nil, result); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if proof.CutCapacity >= proof.RequiredFlow {
		t.Fatalf("cut does not prove infeasibility: %+v", proof)
	}
}

func TestMatchesDirectOracle(t *testing.T) {
	for seed := int64(0); seed < 1000; seed++ {
		nodes, shards, old := randomInstance(seed)
		checkAgainstOracle(t, nodes, shards, old)
	}
}

// Equal objectives are not enough here. Canonicalization must make the entire
// serialized result independent of caller-provided ordering.
func TestPermutationDeterminism(t *testing.T) {
	nodes, shards, old := randomInstance(91)
	baseline, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatal(err)
	}
	want, err := json.Marshal(baseline)
	if err != nil {
		t.Fatal(err)
	}

	random := rand.New(rand.NewSource(17))
	for trial := 0; trial < 200; trial++ {
		shuffledNodes := append([]sf.Node(nil), nodes...)
		shuffledShards := append([]sf.Shard(nil), shards...)
		shuffledOld := clonePlacement(old)
		random.Shuffle(len(shuffledNodes), func(i, j int) {
			shuffledNodes[i], shuffledNodes[j] = shuffledNodes[j], shuffledNodes[i]
		})
		random.Shuffle(len(shuffledShards), func(i, j int) {
			shuffledShards[i], shuffledShards[j] = shuffledShards[j], shuffledShards[i]
		})
		random.Shuffle(len(shuffledOld), func(i, j int) {
			shuffledOld[i], shuffledOld[j] = shuffledOld[j], shuffledOld[i]
		})
		for index := range shuffledOld {
			random.Shuffle(len(shuffledOld[index].Nodes), func(i, j int) {
				shuffledOld[index].Nodes[i], shuffledOld[index].Nodes[j] =
					shuffledOld[index].Nodes[j], shuffledOld[index].Nodes[i]
			})
		}

		result, err := sf.Place(shuffledNodes, shuffledShards, shuffledOld)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		got, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("trial %d produced different output\n got: %s\nwant: %s",
				trial, got, want)
		}
	}
}

func TestVerifierRejectsMutations(t *testing.T) {
	nodes := []sf.Node{
		{ID: "a", Domain: "east", Capacity: 1},
		{ID: "b", Domain: "west", Capacity: 2},
	}
	shards := []sf.Shard{
		{ID: "x", Replicas: 1},
		{ID: "y", Replicas: 2},
	}
	old := sf.Placement{
		{Shard: "x", Nodes: []sf.NodeID{"a"}},
		{Shard: "y", Nodes: []sf.NodeID{"a", "b"}},
	}
	result, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatal(err)
	}
	solution := requireSolution(t, result)

	mutations := map[string]func(sf.Solution) sf.Solution{
		"objective": func(solution sf.Solution) sf.Solution {
			solution = cloneSolution(solution)
			solution.Objective.Moves++
			return solution
		},
		"placement": func(solution sf.Solution) sf.Solution {
			solution = cloneSolution(solution)
			solution.Placement[0].Nodes = nil
			return solution
		},
		"potentials": func(solution sf.Solution) sf.Solution {
			solution = cloneSolution(solution)
			solution.Potentials = solution.Potentials[1:]
			return solution
		},
		"unknown shard": func(solution sf.Solution) sf.Solution {
			solution = cloneSolution(solution)
			solution.Placement = append(solution.Placement, sf.Assignment{
				Shard: "unknown",
				Nodes: []sf.NodeID{"a"},
			})
			return solution
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			if err := sf.Verify(nodes, shards, old, mutate(solution)); err == nil {
				t.Fatal("Verify accepted mutated solution")
			}
		})
	}

	cutNodes := nodes[:1]
	cutShards := []sf.Shard{{ID: "z", Replicas: 2}}
	cutResult, err := sf.Place(cutNodes, cutShards, nil)
	if err != nil {
		t.Fatal(err)
	}
	proof := requireInfeasible(t, cutResult)
	badCapacity := cloneInfeasible(proof)
	badCapacity.CutCapacity++
	if err := sf.Verify(cutNodes, cutShards, nil, badCapacity); err == nil {
		t.Fatal("Verify accepted incorrect cut capacity")
	}
	missingSource := cloneInfeasible(proof)
	for index, vertex := range missingSource.Cut {
		if vertex == 0 {
			missingSource.Cut = append(
				missingSource.Cut[:index],
				missingSource.Cut[index+1:]...,
			)
			break
		}
	}
	if err := sf.Verify(cutNodes, cutShards, nil, missingSource); err == nil {
		t.Fatal("Verify accepted cut without source")
	}
}

// Moving one alpha replica to d trades a move for lower concentration. This
// placement passes structural checks, so only reduced costs can reject it.
func TestVerifierRejectsSuboptimalPlacement(t *testing.T) {
	nodes := []sf.Node{
		{ID: "a", Domain: "rack-1", Capacity: 2},
		{ID: "b", Domain: "rack-2", Capacity: 2},
		{ID: "c", Domain: "rack-3", Capacity: 2},
		{ID: "d", Domain: "rack-4", Capacity: 2},
	}
	shards := []sf.Shard{
		{ID: "alpha", Replicas: 2},
		{ID: "beta", Replicas: 2},
		{ID: "gamma", Replicas: 2},
	}
	old := sf.Placement{
		{Shard: "alpha", Nodes: []sf.NodeID{"a", "b"}},
		{Shard: "beta", Nodes: []sf.NodeID{"a", "c"}},
		{Shard: "gamma", Nodes: []sf.NodeID{"b", "c"}},
	}
	result, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatal(err)
	}
	solution := requireSolution(t, result)
	want := sf.Cost{Moves: 0, Concentration: 12}
	if solution.Objective != want {
		t.Fatalf("objective is %+v, want %+v", solution.Objective, want)
	}

	suboptimal := cloneSolution(solution)
	suboptimal.Placement = sf.Placement{
		{Shard: "alpha", Nodes: []sf.NodeID{"b", "d"}},
		{Shard: "beta", Nodes: []sf.NodeID{"a", "c"}},
		{Shard: "gamma", Nodes: []sf.NodeID{"b", "c"}},
	}
	suboptimal.Objective = sf.Cost{Moves: 1, Concentration: 10}
	if err := sf.Verify(nodes, shards, old, suboptimal); err == nil {
		t.Fatal("Verify accepted suboptimal placement")
	}

	// Zero potentials have the right length but leave the reverse arc of every
	// used slot edge with negative reduced cost.
	zeroPotentials := cloneSolution(solution)
	for index := range zeroPotentials.Potentials {
		zeroPotentials.Potentials[index] = sf.Cost{}
	}
	if err := sf.Verify(nodes, shards, old, zeroPotentials); err == nil {
		t.Fatal("Verify accepted zero potentials")
	}
}

// A feasible instance has no proof of infeasibility. Verify must recompute
// each claim rather than trust the certificate's own fields.
func TestVerifierRejectsForgedCuts(t *testing.T) {
	nodes := []sf.Node{{ID: "a", Domain: "east", Capacity: 1}}
	shards := []sf.Shard{{ID: "x", Replicas: 1}}
	result, err := sf.Place(nodes, shards, nil)
	if err != nil {
		t.Fatal(err)
	}
	vertices := len(requireSolution(t, result).Potentials)
	allVertices := make([]int, vertices)
	for index := range allVertices {
		allVertices[index] = index
	}

	// A cut containing the sink has no outgoing edges.
	forged := map[string]sf.Infeasible{
		"sink inside cut": {Cut: allVertices, RequiredFlow: 1, CutCapacity: 0},
		"inflated flow":   {Cut: []int{0}, RequiredFlow: 2, CutCapacity: 1},
		"out of range":    {Cut: []int{0, vertices}, RequiredFlow: 1, CutCapacity: 1},
		"negative index":  {Cut: []int{0, -1}, RequiredFlow: 1, CutCapacity: 1},
	}
	for name, proof := range forged {
		t.Run(name, func(t *testing.T) {
			if err := sf.Verify(nodes, shards, nil, proof); err == nil {
				t.Fatal("Verify accepted forged cut")
			}
		})
	}
}

func TestRemovedOldNodeIsValid(t *testing.T) {
	nodes := []sf.Node{{ID: "live", Domain: "rack", Capacity: 1}}
	shards := []sf.Shard{{ID: "x", Replicas: 1}}
	old := sf.Placement{{Shard: "x", Nodes: []sf.NodeID{"failed"}}}

	result, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatal(err)
	}
	if err := sf.Verify(nodes, shards, old, result); err != nil {
		t.Fatal(err)
	}
	solution := requireSolution(t, result)
	if solution.Objective.Moves != 1 {
		t.Fatalf("moves is %d, want 1", solution.Objective.Moves)
	}
}

// Spending one move can reduce concentration, but lexicographic ordering must
// preserve every old replica before considering that improvement.
func TestLexicographicPriority(t *testing.T) {
	nodes := []sf.Node{
		{ID: "a", Domain: "rack-1", Capacity: 2},
		{ID: "b", Domain: "rack-2", Capacity: 2},
		{ID: "c", Domain: "rack-3", Capacity: 2},
		{ID: "d", Domain: "rack-4", Capacity: 2},
	}
	shards := []sf.Shard{
		{ID: "alpha", Replicas: 2},
		{ID: "beta", Replicas: 2},
		{ID: "gamma", Replicas: 2},
	}
	old := sf.Placement{
		{Shard: "alpha", Nodes: []sf.NodeID{"a", "b"}},
		{Shard: "beta", Nodes: []sf.NodeID{"a", "c"}},
		{Shard: "gamma", Nodes: []sf.NodeID{"b", "c"}},
	}

	result, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatal(err)
	}
	if err := sf.Verify(nodes, shards, old, result); err != nil {
		t.Fatal(err)
	}
	solution := requireSolution(t, result)
	want := sf.Cost{Moves: 0, Concentration: 12}
	if solution.Objective != want {
		t.Fatalf("result is %+v, want objective %+v", result, want)
	}
}

func TestZeroCapacityRepresentsDrain(t *testing.T) {
	nodes := []sf.Node{
		{ID: "drained", Domain: "east", Capacity: 0},
		{ID: "live", Domain: "west", Capacity: 1},
	}
	shards := []sf.Shard{{ID: "x", Replicas: 1}}
	old := sf.Placement{{Shard: "x", Nodes: []sf.NodeID{"drained"}}}

	result, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatal(err)
	}
	if err := sf.Verify(nodes, shards, old, result); err != nil {
		t.Fatal(err)
	}
	solution := requireSolution(t, result)
	if solution.Placement[0].Nodes[0] != "live" {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestRejectsMalformedInput(t *testing.T) {
	tests := []struct {
		name   string
		nodes  []sf.Node
		shards []sf.Shard
		old    sf.Placement
	}{
		{
			name: "duplicate node",
			nodes: []sf.Node{
				{ID: "a", Domain: "east", Capacity: 1},
				{ID: "a", Domain: "west", Capacity: 1},
			},
		},
		{
			name:  "negative capacity",
			nodes: []sf.Node{{ID: "a", Domain: "east", Capacity: -1}},
		},
		{
			name: "duplicate shard",
			shards: []sf.Shard{
				{ID: "x", Replicas: 1},
				{ID: "x", Replicas: 1},
			},
		},
		{
			name:   "duplicate old shard",
			nodes:  []sf.Node{{ID: "a", Domain: "east", Capacity: 1}},
			shards: []sf.Shard{{ID: "x", Replicas: 1}},
			old: sf.Placement{
				{Shard: "x", Nodes: []sf.NodeID{"a"}},
				{Shard: "x", Nodes: []sf.NodeID{"a"}},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := sf.Place(test.nodes, test.shards, test.old); err == nil {
				t.Fatal("Place accepted malformed input")
			}
		})
	}
}

func TestUnknownOldShardIsIgnored(t *testing.T) {
	nodes := []sf.Node{{ID: "a", Domain: "east", Capacity: 1}}
	shards := []sf.Shard{{ID: "x", Replicas: 1}}
	old := sf.Placement{
		{Shard: "retired", Nodes: []sf.NodeID{"gone"}},
		{Shard: "x", Nodes: []sf.NodeID{"gone"}},
	}
	result, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatal(err)
	}
	solution := requireSolution(t, result)
	if solution.Objective.Moves != 1 {
		t.Fatalf("moves is %d, want 1", solution.Objective.Moves)
	}
}

func TestRejectsOversizedNetwork(t *testing.T) {
	const count = 1025
	nodes := make([]sf.Node, count)
	shards := make([]sf.Shard, count)
	for index := 0; index < count; index++ {
		nodes[index] = sf.Node{
			ID:       sf.NodeID(fmt.Sprintf("n%04d", index)),
			Domain:   "rack",
			Capacity: 1,
		}
		shards[index] = sf.Shard{
			ID:       sf.ShardID(fmt.Sprintf("s%04d", index)),
			Replicas: 1,
		}
	}
	if _, err := sf.Place(nodes, shards, nil); err == nil {
		t.Fatal("Place accepted a network above its explicit edge limit")
	}
}

func FuzzPlaceMatchesOracle(f *testing.F) {
	for _, seed := range []int64{0, 1, 7, 42, 999} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, seed int64) {
		nodes, shards, old := randomInstance(seed)
		checkAgainstOracle(t, nodes, shards, old)
	})
}

// Verify establishes that the returned placement is valid. The oracle then
// needs to compare only feasibility and the optimum, since several placements
// may attain the same cost.
func checkAgainstOracle(
	t *testing.T,
	nodes []sf.Node,
	shards []sf.Shard,
	old sf.Placement,
) {
	t.Helper()
	result, err := sf.Place(nodes, shards, old)
	if err != nil {
		t.Fatalf("Place: %v\nnodes=%+v\nshards=%+v\nold=%+v",
			err, nodes, shards, old)
	}
	if err := sf.Verify(nodes, shards, old, result); err != nil {
		t.Fatalf("Verify: %v\nnodes=%+v\nshards=%+v\nold=%+v\nresult=%+v",
			err, nodes, shards, old, result)
	}
	want, feasible := directOracle(nodes, shards, old)
	solution, solved := result.(sf.Solution)
	if feasible != solved {
		t.Fatalf("feasibility differs: solver=%t oracle=%t\nnodes=%+v\nshards=%+v",
			solved, feasible, nodes, shards)
	}
	if feasible && solution.Objective != want {
		t.Fatalf("objective is %+v, want %+v\nnodes=%+v\nshards=%+v\nold=%+v",
			solution.Objective, want, nodes, shards, old)
	}
}

// directOracle enumerates placements without sharing graph, residual, or
// objective-comparison helpers with the solver.
func directOracle(
	nodes []sf.Node,
	shards []sf.Shard,
	old sf.Placement,
) (sf.Cost, bool) {
	oldNodes := make(map[sf.ShardID]map[sf.NodeID]bool, len(old))
	for _, assignment := range old {
		oldNodes[assignment.Shard] = make(map[sf.NodeID]bool, len(assignment.Nodes))
		for _, node := range assignment.Nodes {
			oldNodes[assignment.Shard][node] = true
		}
	}
	loads := make([]int, len(nodes))
	var best *sf.Cost
	var placeShard func(int, int64)

	placeShard = func(shardIndex int, moves int64) {
		if shardIndex == len(shards) {
			cost := sf.Cost{Moves: moves}
			for _, load := range loads {
				cost.Concentration += int64(load * load)
			}
			if best == nil ||
				cost.Moves < best.Moves ||
				cost.Moves == best.Moves &&
					cost.Concentration < best.Concentration {
				copy := cost
				best = &copy
			}
			return
		}

		usedDomains := make(map[string]bool)
		var choose func(int, int, int64)
		choose = func(start, remaining int, nextMoves int64) {
			if remaining == 0 {
				placeShard(shardIndex+1, nextMoves)
				return
			}
			for node := start; node < len(nodes); node++ {
				if loads[node] == nodes[node].Capacity ||
					usedDomains[nodes[node].Domain] {
					continue
				}
				loads[node]++
				usedDomains[nodes[node].Domain] = true
				addedMove := int64(0)
				if !oldNodes[shards[shardIndex].ID][nodes[node].ID] {
					addedMove = 1
				}
				choose(node+1, remaining-1, nextMoves+addedMove)
				delete(usedDomains, nodes[node].Domain)
				loads[node]--
			}
		}
		choose(0, shards[shardIndex].Replicas, moves)
	}

	placeShard(0, 0)
	if best == nil {
		return sf.Cost{}, false
	}
	return *best, true
}

// Generated instances stay small enough for exhaustive enumeration while
// varying capacity, failure domains, replica counts, and removed nodes.
func randomInstance(seed int64) ([]sf.Node, []sf.Shard, sf.Placement) {
	random := rand.New(rand.NewSource(seed))
	domainCount := 1 + random.Intn(3)
	nodeCount := 1 + random.Intn(5)
	nodes := make([]sf.Node, nodeCount)
	for index := range nodes {
		nodes[index] = sf.Node{
			ID:       sf.NodeID(fmt.Sprintf("n%d", index)),
			Domain:   fmt.Sprintf("d%d", random.Intn(domainCount)),
			Capacity: random.Intn(3),
		}
	}

	shardCount := 1 + random.Intn(4)
	shards := make([]sf.Shard, shardCount)
	old := make(sf.Placement, shardCount)
	for index := range shards {
		replicas := 1 + random.Intn(3)
		shards[index] = sf.Shard{
			ID:       sf.ShardID(fmt.Sprintf("s%d", index)),
			Replicas: replicas,
		}
		old[index].Shard = shards[index].ID
		pool := make([]sf.NodeID, 0, nodeCount+4)
		for _, node := range nodes {
			pool = append(pool, node.ID)
		}
		// Removed nodes remain valid in an old placement, so generated cases
		// include them to exercise drain and replacement behavior.
		for gone := 0; gone < 4; gone++ {
			pool = append(pool, sf.NodeID(fmt.Sprintf("gone%d", gone)))
		}
		random.Shuffle(len(pool), func(i, j int) {
			pool[i], pool[j] = pool[j], pool[i]
		})
		old[index].Nodes = append([]sf.NodeID(nil), pool[:replicas]...)
	}
	return nodes, shards, old
}

func clonePlacement(placement sf.Placement) sf.Placement {
	clone := make(sf.Placement, len(placement))
	for index, assignment := range placement {
		clone[index] = sf.Assignment{
			Shard: assignment.Shard,
			Nodes: append([]sf.NodeID(nil), assignment.Nodes...),
		}
	}
	return clone
}

func cloneSolution(solution sf.Solution) sf.Solution {
	return sf.Solution{
		Placement:  clonePlacement(solution.Placement),
		Objective:  solution.Objective,
		Potentials: append([]sf.Cost(nil), solution.Potentials...),
	}
}

func cloneInfeasible(proof sf.Infeasible) sf.Infeasible {
	return sf.Infeasible{
		Cut:          append([]int(nil), proof.Cut...),
		RequiredFlow: proof.RequiredFlow,
		CutCapacity:  proof.CutCapacity,
	}
}

func requireSolution(t *testing.T, result sf.Result) sf.Solution {
	t.Helper()
	solution, ok := result.(sf.Solution)
	if !ok {
		t.Fatalf("got %T, want shardflow.Solution", result)
	}
	return solution
}

func requireInfeasible(t *testing.T, result sf.Result) sf.Infeasible {
	t.Helper()
	proof, ok := result.(sf.Infeasible)
	if !ok {
		t.Fatalf("got %T, want shardflow.Infeasible", result)
	}
	return proof
}
