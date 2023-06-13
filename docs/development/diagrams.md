# PlantUML Diagram for Agent Message Handling Cascade

```plantuml
@startuml
state "Agent" as agent_outer {
  state "Message arrives" as Message
  state "Done?" as Done
  state "LLM" as LLM
  state "Tool" as Tool
  state "Human" as Human
  state "Sub-Agent" as agent_inner
  state "Return Null to parent" as Return

  Message --> Done
  Done -down-> LLM : No
  Done --> ReturnResult : Yes
  LLM -down-> Tool : No
  LLM --> Message : Yes
  Tool -down-> Human : No
  Tool --> Message : Yes
  Human -down-> agent_inner : No
  Human --> Message : Yes
  agent_inner -down-> Return : No
  agent_inner --> Message : Yes
  
}
state "Return result to parent" as ReturnResult

@enduml
```

# Mermaid diagram for Agent message handling

```mermaid
graph LR
    subgraph Agent Message Handling Cascade, Delegating to Agent-1
        MSG[Message] --> Done{Done?}
        Done -- No --> LLM{LLM}
        Done -- Yes --> ReturnDone[Return result to Parent Task]
        LLM -- No --> TOOL{Tool}
        LLM -- Yes --> MSG
        TOOL -- No --> HUMAN{Human}
        TOOL -- Yes --> MSG
        HUMAN -- No --> AGENT{Agent-1 Task}
        HUMAN -- Yes --> MSG
        AGENT -- No --> Return[Return Null to parent Task]
        AGENT -- Yes --> MSG
    end

```

# PlantUML for Agent Structure

```plantuml
@startuml
package "Agent" {
  [LLM]
  [Vector-store]
  [Tools]
}
@enduml
```

# Docker Expert 3-agent communication: nested delegation

```mermaid
 sequenceDiagram
      participant DockerExpert  
      participant Helper  
      participant Coder  
  
      DockerExpert->>Helper: Q1  
      Helper->>Coder: q_1  
      Coder->>Helper: a_1  
      Helper->>Coder: q_2  
      Coder->>Helper: a_2  
      Helper->>Coder: q_3  
      Coder->>Helper: a_3  
      Helper->>DockerExpert: A1
```

```plantuml
@startuml
!theme sketchy-outline
participant Agent1
participant Agent2
participant Agent3

Agent1 -> Agent2: Question
Agent2 -> Agent3: q_1
Agent3 -> Agent2: a_1
Agent2 -> Agent3: q_2
Agent3 -> Agent2: a_2
Agent2 -> Agent3: q_3
Agent3 -> Agent2: a_3
Agent2 -> Agent1: Answer
@enduml
```

```plantuml
@startuml
!theme bluegray
participant DockerExpert
participant Helper
participant Coder

DockerExpert -> Helper: Which python version?
Helper -> Coder: Is there a setup.py file?
Coder -> Helper: yes
Helper -> Coder: What python version is mentioned there?
Coder -> Helper: No python version found.
Helper -> Coder: is there a python version in the pyproject.toml file?
Coder -> Helper: yes - it is ^3.10
Helper -> DockerExpert: Python ^3.10
@enduml

```
