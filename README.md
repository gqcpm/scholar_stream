# Scholar Stream: Autonomous Research Agent System

An **autonomous multi-agent system** for automated research paper discovery, analysis, and report generation using **agentic AI** principles. This project demonstrates advanced **agent orchestration**, **state management**, and **iterative refinement** capabilities.

## 🎯 Overview

This repository implements a **production-ready research agent** that autonomously:
- **Decomposes complex research tasks** into actionable sub-queries
- **Searches and retrieves** relevant academic papers from arXiv
- **Synthesizes information** into structured research reports
- **Self-critiques and iteratively refines** outputs for quality assurance

Built with **LangGraph** for **multi-agent orchestration** and fine-tuned **Qwen3-14B** models with **LoRA adapters** for enhanced reasoning capabilities.

## 🏗️ Architecture

### Multi-Agent System Design

The system implements a **stateful agent workflow** using **LangGraph** with four specialized agents:

#### 1. **Planner Agent** (Task Decomposition)
- **Autonomous task breakdown**: Converts high-level research questions into specific, actionable search queries
- **Strategic planning**: Generates 3-5 targeted queries optimized for academic paper discovery
- **Prompt engineering**: Uses system prompts to ensure structured, parseable outputs

#### 2. **Researcher Agent** (Information Retrieval)
- **Autonomous paper discovery**: Searches arXiv using generated queries
- **Relevance-based ranking**: Leverages arXiv's relevance sorting for optimal results
- **Content extraction**: Retrieves titles and abstracts for downstream processing

#### 3. **Writer Agent** (Content Synthesis)
- **Autonomous report generation**: Synthesizes multiple research sources into coherent reports
- **Context-aware generation**: Uses fine-tuned model with **thinking/reasoning capabilities**
- **Structured output**: Produces well-organized, citation-ready research summaries

#### 4. **Critic Agent** (Quality Assurance & Self-Reflection)
- **Autonomous quality control**: Evaluates drafts for logical errors, hallucinations, and clarity
- **Iterative refinement loop**: Provides specific, actionable feedback
- **Conditional routing**: Determines whether to continue refinement or finalize output

### State Management

The system uses **TypedDict-based state management** to track:
- Original task and decomposed plan
- Retrieved research content
- Draft iterations and revision history
- Critique feedback and quality metrics
- Loop control (max revisions, approval status)

### Agent Orchestration

**LangGraph** enables:
- **Conditional routing**: Dynamic workflow paths based on agent decisions
- **State persistence**: Seamless data flow between agents
- **Error handling**: Robust failure recovery and graceful degradation
- **Streaming execution**: Real-time progress monitoring

## 🤖 Model Fine-Tuning

### Custom Reasoning Capabilities

The system uses a **fine-tuned Qwen3-14B** model with:
- **LoRA (Low-Rank Adaptation)**: Efficient parameter-efficient fine-tuning
- **4-bit quantization**: Memory-efficient inference with minimal quality loss
- **Thinking/reasoning tokens**: Enhanced chain-of-thought capabilities
- **Unsloth optimization**: 2-3x faster training and inference

### Training Pipeline

1. **Base Model**: `unsloth/Qwen3-14B` with 4-bit quantization
2. **LoRA Adapters**: Fine-tuned for research synthesis and critical thinking
3. **Adapter Loading**: Dynamic adapter loading for specialized capabilities
4. **Inference Optimization**: FastLanguageModel for production-ready deployment

## 🔧 Key Technologies

### Agentic AI Stack
- **LangGraph**: Multi-agent orchestration and workflow management
- **LangChain Core**: Message handling and prompt management
- **State Graphs**: TypedDict-based state management

### LLM Infrastructure
- **Unsloth**: Efficient fine-tuning and inference optimization
- **Qwen3-14B**: Base model with reasoning capabilities
- **LoRA Adapters**: Parameter-efficient fine-tuning
- **4-bit Quantization**: Memory-efficient deployment

### Research Tools
- **arXiv API**: Academic paper discovery and retrieval
- **Python TypedDict**: Type-safe state management

## 📊 Agent Workflow

```
User Query
    ↓
[Planner Agent] → Task Decomposition → Search Queries
    ↓
[Researcher Agent] → arXiv Search → Paper Abstracts
    ↓
[Writer Agent] → Content Synthesis → Draft Report
    ↓
[Critic Agent] → Quality Assessment → Critique
    ↓
    ├─→ [Approved] → Final Report
    └─→ [Needs Revision] → Loop back to Writer (max 3 iterations)
```

## 🚀 Features

### Autonomous Operation
- **Zero-shot task decomposition**: No predefined templates required
- **Self-directed research**: Agents autonomously determine search strategies
- **Iterative self-improvement**: Built-in critique and refinement loops

### Production-Ready
- **Error handling**: Robust parsing and fallback mechanisms
- **State persistence**: Maintains context across agent transitions
- **Streaming execution**: Real-time progress updates
- **Memory efficiency**: 4-bit quantization for scalable deployment

### Advanced Capabilities
- **Chain-of-thought reasoning**: Enhanced thinking capabilities via fine-tuning
- **Multi-source synthesis**: Combines information from multiple papers
- **Quality assurance**: Built-in hallucination detection and critique

## 📁 Repository Structure

```
scholar_stream/
├── research_agent.ipynb          # Main multi-agent system implementation
├── trainer_qwen3_14b.ipynb       # Base model fine-tuning pipeline
├── trainer_thinking.ipynb        # Reasoning/thinking capability training
├── lora_adapters/                 # Trained LoRA adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── README.md                      # This file
```

## 🎓 Use Cases

- **Academic Research**: Automated literature reviews and research synthesis
- **Technology Intelligence**: Tracking latest developments in specific domains
- **Competitive Analysis**: Monitoring research trends and breakthroughs
- **Knowledge Discovery**: Autonomous exploration of research landscapes

## 🔑 Key Agentic AI Concepts Demonstrated

- ✅ **Multi-Agent Systems**: Coordinated agents with specialized roles
- ✅ **Agent Orchestration**: LangGraph-based workflow management
- ✅ **State Management**: TypedDict-based stateful agent communication
- ✅ **Task Decomposition**: Autonomous breakdown of complex queries
- ✅ **Iterative Refinement**: Self-critique and improvement loops
- ✅ **Conditional Routing**: Dynamic workflow paths based on agent decisions
- ✅ **Autonomous Decision Making**: Agents make independent choices
- ✅ **Self-Reflection**: Built-in quality assessment mechanisms
- ✅ **Chain-of-Thought Reasoning**: Enhanced reasoning via fine-tuning
- ✅ **Parameter-Efficient Fine-Tuning**: LoRA adapters for specialized capabilities
- ✅ **Memory-Efficient Deployment**: 4-bit quantization for scalability

## 🛠️ Setup & Usage

### Prerequisites
```bash
pip install langgraph langchain_core arxiv unsloth
```

### Running the Research Agent

1. **Load Fine-Tuned Model**:
   ```python
   from unsloth import FastLanguageModel
   
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/Qwen3-14B",
       max_seq_length=2048,
       load_in_4bit=True,
   )
   model.load_adapter("path/to/lora_adapters")
   FastLanguageModel.for_inference(model)
   ```

2. **Initialize Agent System**:
   ```python
   initial_state = {
       "task": "Your research question here",
       "max_revisions": 3,
       "revision_number": 0,
   }
   
   for output in app.stream(initial_state):
       # Process agent outputs
   ```

## 📈 Performance Optimizations

- **Unsloth Integration**: 2-3x faster training and inference
- **4-bit Quantization**: ~75% memory reduction with minimal accuracy loss
- **LoRA Adapters**: Efficient fine-tuning with <1% of base model parameters
- **Streaming Execution**: Real-time progress without blocking

## 🔬 Research & Development

This project demonstrates:
- **Production-grade agentic AI** implementation
- **End-to-end autonomous research** workflows
- **Fine-tuning for specialized reasoning** capabilities
- **Scalable multi-agent architectures**

## 📝 License

This project is open source and available for research and educational purposes.

---

**Built with**: LangGraph • Unsloth • Qwen3-14B • LoRA • Agentic AI
