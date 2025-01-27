from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith.schemas import Run, Example
from langchain.output_parsers import RegexParser

class Evaluator:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            temperature=temperature,
            verbose=True,
            model=model_name
        )
        self.output_parser = RegexParser(
            regex=r"\[\[(\d+)\]\]",
            output_keys=["rating"],
            default_output_key="rating"
        )

    def _extract_llm_answer(self, run: Run) -> str:
        """Extract story content from run outputs"""
        outputs = run.outputs or {}
        if isinstance(outputs, dict):
            output = outputs.get("output", "")
            response = output.get("response", "")
            answer = response.get("answer", "")
            return answer if hasattr(answer, 'answer') else str(answer)
        return str(outputs)

    def _extract_example_output(self, example: Example) -> str:
        """Extract input parameters from example"""
        output = example.outputs or {}
        return output.get("output", "")

    def _extract_example_correct_answer(self, example: Example) -> str:
        """Extract correct answer from example"""
        input = example.inputs or {}
        return input.get("correctAnswer", "")
    
    def evaluate_answer_match(self, run: Run, example: Example) -> dict:
        """Evaluates wether the LLM answer matches the student's answer."""
        llm_answer = self._extract_llm_answer(run)
        if not isinstance(llm_answer, str):
            return {"key": "match", "score": 0, "error": "Invalid prediction type"}
        
        example_output = self._extract_example_output(example)
        is_match = llm_answer == example_output
        return {"key": "answer_match", "score": int(is_match)}
    
    def evaluate_llm_correct(self, run: Run, example: Example) -> dict:
        """Evaluates wether the LLM answer matches the correct answer."""
        llm_answer = self._extract_llm_answer(run)
        if not isinstance(llm_answer, str):
            return {"key": "correctness", "score": 0, "error": "Invalid prediction type"}
        
        example_correct_answer = self._extract_example_correct_answer(example)
        is_correct = llm_answer == example_correct_answer
        return {"key": "llm_correct", "score": int(is_correct)}
    
    def evaluate_correct_match(self, run: Run, example: Example) -> dict:
        """Evaluates wether the LLM correctness matches the student's correctness."""
        llm_answer = self._extract_llm_answer(run)
        if not isinstance(llm_answer, str):
            return {"key": "correctness", "score": 0, "error": "Invalid prediction type"}

        student_answer = self._extract_example_output(example)

        correct_answer = self._extract_example_correct_answer(example)

        is_student_correct = student_answer == correct_answer
        is_llm_correct = llm_answer == correct_answer

        is_correct = is_student_correct == is_llm_correct
        return {"key": "correct_match", "score": int(is_correct)}
