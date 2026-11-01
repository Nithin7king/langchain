Here are the notes and code explanations from the video **"Chains in LangChain | Generative AI using LangChain | Video 7 | CampusX"**:

### 1. What are Chains in LangChain?

When building an LLM application, you typically execute multiple steps sequentially. For example:

1. Receive a topic from the user.
2. Format that topic into a prompt using `PromptTemplate`.
3. Send the prompt to the language model.
4. Take the model's raw output and extract the relevant text using an `OutputParser`.

Manually invoking each of these steps (`prompt.invoke()`, `model.invoke()`, `parser.invoke()`) creates a lot of boilerplate code. **Chains** solve this by allowing you to connect these components into a single **pipeline**. The output of one component automatically becomes the input to the next.

LangChain uses a declarative syntax called **LCEL (LangChain Expression Language)**, which utilizes the pipe operator (`|`) to link components together cleanly.

### 2. Creating a Simple Sequential Chain

The simplest form of a chain is one that runs in a straight line: `Prompt -> Model -> Parser`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Define the components
template = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)
model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

# 2. Build the Chain using LCEL (the pipe operator)
chain = template | model | parser

# 3. Execute the Chain
# The input dictionary goes to the template, the formatted prompt goes to the model, 
# and the raw output goes to the parser.
result = chain.invoke({"topic": "Cricket"})

print(result)

# Optional: Visualize the chain architecture
chain.get_graph().print_ascii()

```

### 3. Creating a Multi-Step Sequential Chain

You can chain multiple LLM calls together. For example, asking an LLM to write a detailed report, and then passing that *exact* report back into the LLM (or a different LLM) to summarize it.

```python
# 1. Define Components
prompt_1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)
prompt_2 = PromptTemplate(
    template="Generate a 5-pointer summary from the following text: \n{text}",
    input_variables=["text"]
)
model = ChatOpenAI()
parser = StrOutputParser()

# 2. Build a multi-step chain
# Output of parser 1 (the string report) becomes the 'text' input for prompt 2
chain = prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({"topic": "Unemployment in India"})
print(result)

```

### 4. Creating a Parallel Chain

Sometimes you need to execute tasks simultaneously to save time. For example, given a long text document, you might want one model to generate summary notes and another model to generate a quiz based on that text *at the same time*.

To do this, LangChain provides `RunnableParallel`. You pass it a dictionary where the keys are the output variable names, and the values are the chains you want to run simultaneously.

```python
from langchain_core.runnables import RunnableParallel

# Assume prompt_notes and prompt_quiz take {text} as input
chain_notes = prompt_notes | model_1 | parser
chain_quiz = prompt_quiz | model_2 | parser

# 1. Create a Parallel Chain
# This will execute both chains simultaneously and return a dictionary: 
# {"notes": "...", "quiz": "..."}
parallel_chain = RunnableParallel(
    notes=chain_notes,
    quiz=chain_quiz
)

# 2. Create a Merging Chain
# Assume prompt_merge takes {notes} and {quiz} as inputs and merges them
merge_chain = prompt_merge | model_1 | parser

# 3. Combine them into a final chain
final_chain = parallel_chain | merge_chain

result = final_chain.invoke({"text": "Your long document text here..."})
print(result)

```

### 5. Creating a Conditional Chain (Branching)

Sometimes you need `if/else` logic in your pipelines. For example, if a user submits feedback, you might use a Pydantic Output Parser to classify it strictly as "Positive" or "Negative". Based on that classification, you want to route the execution to a different prompt (one that thanks the user, or one that apologizes).

To do this, LangChain provides `RunnableBranch`. It takes a series of tuples containing `(condition, chain_to_execute)`.

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Assume classify_chain outputs a Pydantic object with a .sentiment attribute ("positive" or "negative")
# Assume positive_chain and negative_chain are already defined

# 1. Create the Branching Logic
branch_chain = RunnableBranch(
    # If the condition (lambda function) is True, execute positive_chain
    (lambda x: x.sentiment == "positive", positive_chain),
    
    # Else If the condition is True, execute negative_chain
    (lambda x: x.sentiment == "negative", negative_chain),
    
    # Else (Default fallback)
    RunnableLambda(lambda x: "Could not classify sentiment.")
)

# 2. Combine the classifier and the branch
final_chain = classify_chain | branch_chain

result = final_chain.invoke({"feedback": "This is a terrible phone."})
print(result) # Will output an apology based on the negative_chain execution

```

---

To see the instructor code and explain these concepts step-by-step, check out [Chains in LangChain | Generative AI using LangChain | Video 7 | CampusX](https://www.youtube.com/watch?v=5hjrPILA3-8). This video is a great resource for understanding how to combine LangChain components using LangChain Expression Language (LCEL).
