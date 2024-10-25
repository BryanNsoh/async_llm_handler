## Guide to Using `llm_api_handler.py`

This guide covers everything needed to use `llm_api_handler.py`, from the available models to advanced structured output handling with Pydantic models. Users can rely on this documentation to navigate each function without needing to examine the source code.

---

### Table of Contents

1. **Supported Models**
2. **Key Functionalities**
   - Regular Mode
   - Async Batch Mode
   - OpenAI Batch Mode
   - Pydantic Structured Outputs
3. **Prompt Engineering and JSON Schema Usage**
4. **Practical Examples for Different Contexts**
5. **Advanced LLM Integration Patterns**

---

### 1. Supported Models

The handler supports the following models, each varying in cost, speed, and processing power:

- **claude-3-5-sonnet-20241022**
- **gpt-4o**
- **gemini-1.5-pro-002**
- **gpt-4o-mini**
- **gemini-1.5-flash-002**
- **gemini-1.5-flash-8b**

---

### 2. Key Functionalities

The handler provides **three main processing modes**: Regular, Async Batch, and OpenAI Batch. Each mode serves different processing requirements and is compatible with specific models.

#### **Regular Mode**

Handles a single prompt for all supported models, delivering a direct response.

```python
result = await handler.process(
    prompts="Explain how AI impacts sustainable energy.",
    model="gpt-4o",
    temperature=0.5,
    mode="regular"
)
print(result)  # Direct response output
```

#### **Async Batch Mode**

Supports concurrent processing of multiple prompts asynchronously. Works with all models, suitable for rapid and scalable request handling.

```python
prompts = [
    "Describe the role of AI in agriculture.",
    "Explain the principles of neural networks.",
    "What are the economic effects of machine learning?"
]
batch_result = await handler.process(
    prompts=prompts,
    model="gemini-1.5-flash-002",
    mode="async_batch",
    response_format=CustomPydanticModel
)
for item in batch_result.results:
    print(item.response)
```

#### **OpenAI Batch Mode**

**Only available for OpenAI models** (`gpt-4o`, `gpt-4o-mini`). This mode leverages OpenAI’s batch API, making it suitable for handling larger volumes of prompts in a cost-effective manner.

```python
batch_result = await handler.process(
    prompts=prompts,
    model="gpt-4o-mini",
    mode="openai_batch",
    output_dir="batch_outputs"
)
```

> **Note**: Attempting OpenAI Batch Mode on non-OpenAI models (e.g., `claude-3-5-sonnet-20241022` or `gemini-1.5-pro-002`) will cause errors. Ensure model compatibility before using this mode.

---

### 3. Prompt Engineering and JSON Schema Usage

Incorporating JSON schemas into prompts ensures that models produce structured, predictable outputs, particularly when working with Pydantic models and nested JSON.

1. **Retrieve JSON Schema**: Extract the JSON schema from any Pydantic model you wish to use for structured outputs.

   ```python
   schema = CustomResponseModel.model_json_schema()
   ```

2. **Integrate Schema into Prompt**:

   ```python
   prompt = f"""
   Please provide the response exactly in the following format: {schema}
   Question: "What are the societal impacts of autonomous vehicles?"
   """
   result = await handler.process(
       prompts=prompt,
       model="gemini-1.5-pro-002",
       response_format=CustomResponseModel
   )
   ```

---

### 4. Practical Examples for Different Contexts

#### **Scientific Data and Structured JSON Output Example**

For structured outputs, especially with scientific data, `claude-3-5-sonnet-20241022` and `gemini-1.5-pro-002` can be used effectively.

1. **Define the Pydantic Model**:

   ```python
   class ExperimentData(BaseModel):
       experiment_name: str
       metrics: Dict[str, float]
       conclusions: List[str]

   prompt = "Summarize findings from the recent AI experiment on crop growth."

   response = await handler.process(
       prompts=prompt,
       model="claude-3-5-sonnet-20241022",
       response_format=ExperimentData
   )
   ```

2. **Nested Pydantic Models Example**:

   ```python
   class MetricDetail(BaseModel):
       metric_name: str
       value: float
       unit: str

   class ExperimentSummary(BaseModel):
       title: str
       date: str
       metrics: List[MetricDetail]

   prompt = "Provide details of the recent environmental impact study."
   response = await handler.process(
       prompts=prompt,
       model="gemini-1.5-pro-002",
       response_format=ExperimentSummary
   )
   ```

> **Best Practice**: Avoid using inline dictionaries or constraints directly in Pydantic fields. Use separate nested models to enforce clarity and structure.

#### **Iterative Processing and Feedback Loops**

For iterative tasks that require quicker feedback, `gemini-1.5-flash-8b` is a cost-effective option.

```python
prompts = [
    "What are the basic principles of AI?",
    "List notable events in the history of machine learning."
]
response = await handler.process(
    prompts=prompts,
    model="gemini-1.5-flash-8b",
    mode="async_batch"
)
```

#### **Multi-Layered JSON Output with Error Handling**

For complex, multi-layered JSON outputs or sensitive validation processes, apply retry logic and error handling in the prompt or handler configuration.

```python
class UserModel(BaseModel):
    name: str
    age: int
    occupation: str

prompts = [
    "Describe user Alice, a 29-year-old software engineer.",
    "Provide information about user Bob, a 35-year-old architect."
]

# Batch processing with retries and error handling
result = await handler.process(
    prompts=prompts,
    model="gpt-4o",
    response_format=UserModel,
    max_retries=3
)
for entry in result.results:
    if "response" in entry:
        print(entry["response"])
    else:
        print("Error:", entry["error"])
```

---

### 5. Advanced LLM Integration Patterns

#### **Pattern 1: Nested and Structured Pydantic Models**

Define Pydantic models with clear, nested structures, avoiding inline dictionary constraints.

```python
class SummaryModel(BaseModel):
    highlights: List[str]
    significance: str = Field(description="Explain the importance of the summary")

class AnalysisModel(BaseModel):
    summary: SummaryModel
    keywords: List[str]

prompt = "Analyze the societal effects of climate change."
result = await handler.process(
    prompts=prompt,
    model="gpt-4o-mini",
    response_format=AnalysisModel
)
```

> **Best Practice**: Clearly define required structures in Pydantic models, enforcing structured, validated responses.

#### **Pattern 2: Consistent JSON Response Requests**

Retrieve JSON schemas from Pydantic models and use them to enforce structured JSON outputs directly within prompt construction.

```python
schema = AnalysisModel.model_json_schema()
prompt = f"""
Please respond in this format: {schema}
Question: "What are current AI applications in environmental research?"
"""

response = await handler.process(
    prompts=prompt,
    model="claude-3-5-sonnet-20241022",
    response_format=AnalysisModel
)
```

> **Tip**: Ensure the schema is accurate to avoid mismatches in output structure.

---

### Final Considerations

1. **Model-Specific Capabilities**: Only OpenAI models (`gpt-4o`, `gpt-4o-mini`) support `openai_batch`. Attempting this on other models will result in errors.
2. **Rate Limit Handling**: Use async batch processing to manage higher rate limits with concurrent processing.
3. **Error Handling**: Enable retry and error handling for robustness in complex, structured responses.
4. **Pydantic Model Complexity**: Use nested models for clarity in structured data. Avoid in-line constraints and dictionary fields.
