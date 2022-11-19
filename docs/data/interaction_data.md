# Interaction Data

::: flecs.data.interaction_data.InteractionData
    rendering:
        show_source: false
        heading_level: 2
        show_root_heading: true
        show_signature_annotations: false
        show_signature: true
        members_order: source

## Types

### Base types

```
NodeIdx = int
EdgeIdx = Tuple[int, int]
NodeType = str
RelationType = str
EdgeType = Tuple[str, str, str]
AttributeName = str
Attribute = Union[str, torch.Tensor]
AttributeList = Union[List[str], torch.Tensor]
```

### Specific Types

```
Data = Dict[AttributeName, Attribute]
NodeData = Dict[NodeIdx, Data]
EdgeData = Dict[EdgeIdx, Data]
SetData = Dict[AttributeName, AttributeList]
NodeSetData = Dict[NodeType, SetData]
EdgeSetData = Dict[EdgeType, SetData]
```