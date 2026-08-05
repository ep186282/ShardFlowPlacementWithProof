package shardflow

// NodeID and ShardID are stable external identifiers.
type (
	NodeID  string
	ShardID string
)

// Node is a placement target with one failure domain and a slot capacity.
type Node struct {
	ID       NodeID
	Domain   string
	Capacity int
}

// Shard identifies a replicated object and its required replica count.
type Shard struct {
	ID       ShardID
	Replicas int
}

// Assignment lists the nodes selected for one shard.
type Assignment struct {
	Shard ShardID
	Nodes []NodeID
}

// Placement is kept as a sorted slice so output does not depend on map order.
type Placement []Assignment

// Solution contains a valid placement and its optimality certificate.
type Solution struct {
	Placement  Placement
	Objective  Cost
	Potentials []Cost
}

// Infeasible contains a source-side cut in canonical vertex order.
type Infeasible struct {
	Cut          []int
	RequiredFlow int64
	CutCapacity  int64
}

// Result is either a Solution or an Infeasible certificate.
type Result interface{ isResult() }

func (Solution) isResult()   {}
func (Infeasible) isResult() {}
