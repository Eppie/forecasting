import asyncio
import os

from ollama_python.endpoints import GenerateAPI
from ollama_python.models.generate import Completion
import psutil
from brave_search_python_client import (
    BraveSearch,
    WebSearchRequest,
)


async def search() -> None:
    """Run various searches using the Brave Search Python Client (see https://brave-search-python-client.readthedocs.io/en/latest/lib_reference.html)."""
    # Initialize the Brave Search Python client, using the API key from the environment
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    bs = BraveSearch(api_key=api_key)

    # Perform a web search
    response = await bs.web(WebSearchRequest(q="Elon Musk Trump Epstein Files claim June 2025"))

    # Print results as JSON

    # Iterate over web hits and render links in markdown
    for _result in response.web.results if response.web else []:
        print(_result)


def get_ollama_response(prompt, model="gemma3:27b") -> Completion:
    """
    Makes a REST call to a locally running Ollama server to get an LLM response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str, optional): The name of the Ollama model to use. Defaults to "gemma3:27b".

    Returns:
        str: The LLM response if stream is False, or a generator yielding chunks of the response if stream is True.
        None: If there's an error connecting to the server or getting a response.  Prints error message to console.
    """

    api = GenerateAPI(base_url="http://localhost:11434/api", model=model)
    result = api.generate(prompt=prompt)
    return result


# if __name__ == '__main__':
#     prompt_text = "What is the capital of France?"
#     response = get_ollama_response(prompt_text)
#     print(response.response)


import json
from typing import List, Dict, Any, Optional

# Assumption: The 'ollama_python' library is installed.
# pip install ollama-python
from ollama_python.endpoints import GenerateAPI


# ----------------------------------------------------------------------------
# 1. LLM Interface (As provided by the user)
# ----------------------------------------------------------------------------

class LLMInterface:
    """An interface for interacting with a Large Language Model."""

    def __init__(self, model: str = "gemma3:27b", base_url: str = "http://localhost:11434/api"):
        """Initializes the LLM API endpoint."""
        self.api = GenerateAPI(base_url=base_url, model=model)

    def query(self, prompt: str) -> str:
        """Sends a prompt to the LLM and returns the generated response."""
        try:
            result = self.api.generate(prompt=prompt)
            return result.response
        except Exception as e:
            print(f"Error querying LLM: {e}")
            # Return an empty string or error message to prevent crashes
            return "Error communicating with LLM."


# ----------------------------------------------------------------------------
# 2. Data Structures for AOT
# ----------------------------------------------------------------------------

class SubQuestion:
    """Represents a single node (a subquestion) in the reasoning graph."""

    def __init__(self, id: int, description: str, answer: str = "", dependencies: Optional[List[int]] = None):
        """
        Initializes a subquestion.

        Args:
            id: A unique integer identifier for the subquestion.
            description: The text of the subquestion.
            answer: The answer to the subquestion, if resolved.
            dependencies: A list of IDs of other subquestions that this one depends on.
        """
        self.id = id
        self.description = description
        self.answer = answer
        self.dependencies = dependencies if dependencies is not None else []

    def to_dict(self) -> Dict[str, Any]:
        """Converts the subquestion to a dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "answer": self.answer,
            "dependencies": self.dependencies
        }


class DAG:
    """Represents a Dependency Directed Acyclic Graph of subquestions."""

    def __init__(self, subquestions: List[SubQuestion]):
        """Initializes the DAG with a list of subquestions."""
        self.subquestions = {sq.id: sq for sq in subquestions}

    def get_independent_subquestions(self) -> List[SubQuestion]:
        """Identifies and returns all subquestions with no dependencies."""
        return [sq for sq in self.subquestions.values() if not sq.dependencies]

    def get_dependent_subquestions(self) -> List[SubQuestion]:
        """Identifies and returns all subquestions with dependencies."""
        return [sq for sq in self.subquestions.values() if sq.dependencies]

    def get_max_path_length(self) -> int:
        """
        Calculates the length of the longest path in the DAG. This is a proxy
        for the reasoning depth.
        """
        memo = {}
        max_depth = 0
        for sq_id in self.subquestions:
            max_depth = max(max_depth, self._calculate_depth(sq_id, memo))
        return max_depth

    def _calculate_depth(self, sq_id: int, memo: Dict[int, int]) -> int:
        """Helper for recursively calculating depth of a node."""
        if sq_id in memo:
            return memo[sq_id]

        sq = self.subquestions[sq_id]
        if not sq.dependencies:
            memo[sq_id] = 1
            return 1

        max_prev_depth = 0
        for dep_id in sq.dependencies:
            if dep_id in self.subquestions:
                max_prev_depth = max(max_prev_depth, self._calculate_depth(dep_id, memo))

        memo[sq_id] = 1 + max_prev_depth
        return memo[sq_id]


# ----------------------------------------------------------------------------
# 3. Prompts for the LLM
# ----------------------------------------------------------------------------

def get_decomposition_prompt(question: str) -> str:
    """Generates the prompt for the decomposition phase."""
    return f"""
You are a reasoning agent. Your task is to break down a complex question into smaller, manageable subquestions.
Analyze the original question below and identify the sequence of subquestions that must be answered to arrive at the final solution.
For each subquestion, identify its dependencies on other subquestions. A subquestion is dependent if it requires the answer of another subquestion to be solved.

Original Question: "{question}"

Please format your response as a single JSON object containing a list of subquestions. Each subquestion should have:
- "id": An integer index, starting from 0.
- "description": The subquestion text.
- "dependencies": A list of integer IDs of the subquestions it depends on. An empty list means it's an independent question.
- "answer": Leave this as an empty string for now.

Example Response Format:
{{
  "subquestions": [
    {{
      "id": 0,
      "description": "What is the value of cos(B) given sin(B) = 3/5?",
      "dependencies": [],
      "answer": ""
    }},
    {{
      "id": 1,
      "description": "Using the Law of Cosines and cos(B) > 0, what is the equation for side length BC (let's call it a1)?",
      "dependencies": [0],
      "answer": ""
    }},
    {{
      "id": 2,
      "description": "Using the Law of Cosines and cos(B) < 0, what is the equation for side length BC (let's call it a2)?",
      "dependencies": [0],
      "answer": ""
    }}
  ]
}}

Now, provide the JSON for the original question.
"""


def get_subquestion_solver_prompt(question: str) -> str:
    """Generates the prompt to solve an independent subquestion."""
    return f"""
You are a precise math and logic solver. Solve the following question directly and provide only the final answer without explanations or extra text.

Question: "{question}"

Answer:
"""


def get_contraction_prompt(original_question: str, independent_solved: List[SubQuestion],
                           dependent_unsolved: List[SubQuestion]) -> str:
    """Generates the prompt for the contraction phase."""
    independent_conditions = "\n".join(
        [f"- '{sq.description}' is known to be '{sq.answer}'." for sq in independent_solved]
    )
    dependent_descriptions = "\n".join(
        [f"- {sq.description}" for sq in dependent_unsolved]
    )

    return f"""
You are a reasoning process optimizer. Your task is to create a new, single, self-contained question that incorporates known information and focuses on the remaining unsolved parts of a problem.

Original Question: "{original_question}"

The following subproblems have been solved and can be treated as known conditions:
{independent_conditions}

The remaining parts of the problem that still need to be solved are:
{dependent_descriptions}

Your task is to synthesize all this information into a single, simplified, and self-contained question that logically follows from the known conditions and addresses the remaining problems. The new question should be solvable on its own.

Please provide only the new, optimized question within <question></question> tags.
"""


def get_final_solver_prompt(question: str) -> str:
    """Generates a prompt for a final, comprehensive answer."""
    return f"""
You are an expert problem solver. Provide a step-by-step solution to the following question. Explain your reasoning clearly and enclose the final numerical answer within <answer></answer> tags.

Question: "{question}"
"""


# ----------------------------------------------------------------------------
# 4. Atom of Thought (AOT) Implementation
# ----------------------------------------------------------------------------

class Aot:
    """Implements the Atom of Thoughts (AOT) reasoning framework."""

    def __init__(self, llm: LLMInterface, max_iterations: int = 3):
        """
        Initializes the Aot instance.

        Args:
            llm: An instance of a class that adheres to the LLMInterface.
            max_iterations: The maximum number of decomposition-contraction cycles.
        """
        self.llm = llm
        self.max_iterations = max_iterations

    def _decompose(self, question: str) -> Optional[DAG]:
        """Decomposes the question into a dependency graph."""
        print("\n--- DECOMPOSITION PHASE ---")
        prompt = get_decomposition_prompt(question)
        response_json_str = self.llm.query(prompt)

        try:
            # Clean up the response to extract only the JSON part
            json_start = response_json_str.find('{')
            json_end = response_json_str.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise json.JSONDecodeError("No JSON object found", response_json_str, 0)

            clean_json_str = response_json_str[json_start:json_end]
            data = json.loads(clean_json_str)

            subquestions = [SubQuestion(**sq) for sq in data.get("subquestions", [])]
            if not subquestions:
                return None

            print(f"Decomposed into {len(subquestions)} subquestions.")
            return DAG(subquestions)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error: Failed to parse LLM response into a DAG. Details: {e}")
            print(f"LLM Response:\n{response_json_str}")
            return None

    def _contract(self, dag: DAG, original_question: str) -> Optional[str]:
        """Contracts the DAG into a new, simplified question."""
        print("\n--- CONTRACTION PHASE ---")
        independent_qs = dag.get_independent_subquestions()
        dependent_qs = dag.get_dependent_subquestions()

        if not independent_qs:
            print("No independent subquestions to solve. Contraction skipped.")
            return original_question

        # Solve independent subquestions
        print("Solving independent subquestions...")
        for sq in independent_qs:
            solve_prompt = get_subquestion_solver_prompt(sq.description)
            sq.answer = self.llm.query(solve_prompt).strip()
            print(f"  - Q: {sq.description}\n    A: {sq.answer}")

        if not dependent_qs:
            print("No dependent subquestions remain. The problem is fully decomposed.")
            # Combine solved answers to form the new "question"
            return "\n".join([f"Given {sq.description} is {sq.answer}" for sq in
                              independent_qs]) + f"\nWhat is the final answer to the original question: '{original_question}'?"

        # Contract to form a new question
        print("Contracting to form a new question...")
        contraction_prompt = get_contraction_prompt(original_question, independent_qs, dependent_qs)
        response = self.llm.query(contraction_prompt)

        # Extract content from <question> tags
        try:
            start_tag = "<question>"
            end_tag = "</question>"
            start_index = response.find(start_tag) + len(start_tag)
            end_index = response.find(end_tag)

            if start_index == -1 or end_index == -1:
                print("Warning: Contraction did not produce a valid <question> tag. Using full response.")
                new_question = response.strip()
            else:
                new_question = response[start_index:end_index].strip()

            print(f"New Contracted Question: {new_question}")
            return new_question
        except Exception as e:
            print(f"Error processing contracted question: {e}")
            return None

    def solve(self, question: str) -> str:
        """
        Solves a given question by iteratively applying the AOT process.
        This method implements the main loop from the paper.
        """
        current_question = question
        print(f"Original Question: {current_question}")

        for i in range(self.max_iterations):
            print(f"\n================ ITERATION {i + 1} ================")
            dag = self._decompose(current_question)
            print(dag)

            if dag is None or not dag.get_dependent_subquestions():
                print("\nDecomposition resulted in no further dependent questions. Moving to final solve.")
                break

            new_question = self._contract(dag, current_question)

            if new_question is None or new_question == current_question:
                print("\nContraction failed to simplify the question. Moving to final solve.")
                break

            current_question = new_question

        # Final solve step
        print("\n================ FINAL SOLVE ================")
        final_prompt = get_final_solver_prompt(current_question)
        final_answer = self.llm.query(final_prompt)

        return final_answer


# ----------------------------------------------------------------------------
# 5. Example Usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(search())
    exit()
    # Initialize the LLM interface. Make sure your Ollama server is running.
    llm_interface = LLMInterface(model="gemma3:27b")

    # Initialize the AOT framework
    aot_solver = Aot(llm=llm_interface, max_iterations=2)

    # Define a complex question (from the paper's appendix)
    problem = (
        "For a given constant b > 10, there are two possible triangles ABC "
        "satisfying AB = 10, AC = b, and sin(B) = 3/5. Find the positive "
        "difference between the lengths of side BC in these two triangles."
    )

    # Solve the problem using the AOT technique
    solution = aot_solver.solve(problem)

    print("\n================ FINAL SOLUTION ================")
    print(solution)
