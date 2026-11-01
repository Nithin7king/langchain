Here are the notes and code explanations from the video **"Output Parsers in LangChain | Generative AI using LangChain | Video 6 | CampusX"**:

### 1. What are Output Parsers?

In the previous video, we learned that some LLMs (like OpenAI's GPT models) are fine-tuned to return structured output natively using the `with_structured_output` function. However, many models (especially open-source ones) cannot do this natively.

**Output Parsers** solve this. They are LangChain classes that take the raw, unstructured text response from any LLM and convert it into a structured format (like JSON, CSV, or Pydantic models). They work with *both* models that support structured output and those that do not.

### 2. The Four Main Output Parsers

The video covers the four most commonly used output parsers in LangChain:

#### A. String Output Parser (`StrOutputParser`)

The simplest parser. When an LLM returns a response, it returns an object containing the text along with a lot of metadata (token usage, etc.). The `StrOutputParser` simply extracts the text (`result.content`) into a clean string.
It is extremely useful when building **Chains** (pipelines) where the output of one LLM call needs to be fed directly as a string into the prompt of the next LLM call.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-3.5-turbo")

# 1. Define Templates
template_1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template_2 = PromptTemplate(
    template="Write a 5-line summary on the following text: {text}",
    input_variables=["text"]
)

# 2. Initialize the Parser
parser = StrOutputParser()

# 3. Create a Chain 
# The parser extracts the string from model 1 and feeds it directly into template 2
chain = template_1 | model | parser | template_2 | model | parser

# Execute
result = chain.invoke({"topic": "Black Holes"})
print(result)

```

#### B. JSON Output Parser (`JsonOutputParser`)

This parser forces the LLM to return a JSON object (which Python reads as a dictionary). It does this by generating formatting instructions that you inject into your prompt.
*Drawback:* It does not enforce a specific schema. The LLM decides the structure of the JSON keys and values.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOpenAI(model="gpt-3.5-turbo")
parser = JsonOutputParser()

# Inject format instructions into the prompt
template = PromptTemplate(
    template="Give me the name, age, and city of a fictional person. \n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})

print(result) # Returns a Python Dictionary
print(result['name']) 

```

#### C. Structured Output Parser (`StructuredOutputParser`)

This solves the problem of the JSON parser. It allows you to define a specific schema (keys and descriptions) that the LLM must follow when generating the JSON.
*Drawback:* It enforces a structure, but it **cannot perform data validation** (e.g., if you ask for an integer age, but the LLM returns the string "35 years", the code will not catch the error).

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

model = ChatOpenAI(model="gpt-3.5-turbo")

# 1. Define the Schema using ResponseSchema
response_schemas = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic")
]

# 2. Initialize Parser with the Schema
parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = PromptTemplate(
    template="Give 2 facts about {topic}. \n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "Black Holes"})
print(result)

```

#### D. Pydantic Output Parser (`PydanticOutputParser`)

The most powerful and recommended parser. It uses Pydantic objects to define the schema. Not only does it enforce the structure of the JSON, but it also **performs strict data validation at runtime.** (If the LLM returns a string when an integer is required, Pydantic will catch it, attempt to convert it, or throw an error).

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

model = ChatOpenAI(model="gpt-3.5-turbo")

# 1. Define Schema and Validation using Pydantic
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    # Enforce constraints: Age must be greater than 18
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City the person belongs to")

# 2. Initialize Parser
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age, and city of a fictional {place} person. \n{format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"place": "Sri Lankan"})

print(result) # Returns a validated Pydantic object
print(result.age)

```

---

To see the instructor code and explain these concepts step-by-step, check out [Output Parsers in LangChain | Generative AI using LangChain | Video 6 | CampusX](https://www.google.com/search?q=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DOp6PbJZ5b2Q). This video demonstrates how to use Langchain to extract structured output from language models.
