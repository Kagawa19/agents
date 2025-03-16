"""
Reasoning with Claude.
Provides methods for complex reasoning tasks using Claude.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from multiagent.app.monitoring.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

async def step_by_step_reasoning(
    client: Any,
    problem: str,
    model: str = "claude",
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Solve a problem using step-by-step reasoning.
    
    Args:
        client: Bedrock client
        problem: Problem to solve
        model: Model to use
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Dictionary with steps and solution
    """
    span = None
    if tracer:
        span = tracer.span(
            name="claude_step_reasoning",
            input={"problem_length": len(problem)}
        )
    
    try:
        # Create prompt for step-by-step reasoning
        prompt = f"""
        Solve the following problem step by step. Show your reasoning process clearly.
        
        Problem:
        {problem}
        
        Step-by-step solution:
        """
        
        # Import generate_text
        from multiagent.app.tools.bedrock.claude_generate import generate_text
        
        # Generate response
        response = await generate_text(
            client=client,
            prompt=prompt,
            model=model,
            max_tokens=2000,
            temperature=0.2,
            tracer=tracer
        )
        
        # Extract steps from response
        response_text = response.get("text", "").strip()
        steps = extract_reasoning_steps(response_text)
        
        result = {
            "steps": steps,
            "full_solution": response_text,
            "step_count": len(steps)
        }
        
        if span:
            span.update(output={"step_count": len(steps)})
        
        return result
        
    except Exception as e:
        logger.error(f"Error in step-by-step reasoning: {str(e)}")
        
        if span:
            span.update(output={"error": str(e)})
        
        return {"error": str(e)}

def extract_reasoning_steps(text: str) -> List[str]:
    """
    Extract reasoning steps from text.
    
    Args:
        text: Text containing reasoning steps
        
    Returns:
        List of reasoning steps
    """
    # Try different step patterns
    step_patterns = [
        r"Step (\d+)[:.]\s*(.*?)(?=Step \d+[:.]\s*|$)",  # Step 1: ...
        r"(\d+)\.\s*(.*?)(?=\d+\.\s*|$)",  # 1. ...
        r"(\d+)\)\s*(.*?)(?=\d+\)\s*|$)"   # 1) ...
    ]
    
    for pattern in step_patterns:
        steps = re.findall(pattern, text, re.DOTALL)
        if steps:
            # Extract step content and clean up
            return [step.strip() for _, step in steps]
    
    # If no step pattern found, split by lines or paragraphs
    if "\n\n" in text:
        return [step.strip() for step in text.split("\n\n") if step.strip()]
    else:
        return [step.strip() for step in text.split("\n") if step.strip()]

async def chain_of_thought(
    client: Any,
    question: str,
    context: Optional[str] = None,
    model: str = "claude",
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Answer a question using chain-of-thought reasoning.
    
    Args:
        client: Bedrock client
        question: Question to answer
        context: Optional context information
        model: Model to use
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Dictionary with reasoning and answer
    """
    # Create prompt for chain-of-thought reasoning
    prompt = "Let's think through this step by step.\n\n"
    
    if context:
        prompt += f"""
        Context:
        {context}
        
        """
    
    prompt += f"""
    Question:
    {question}
    
    Thinking:
    """
    
    # Import generate_text
    from multiagent.app.tools.bedrock.claude_generate import generate_text
    
    # Generate thinking process
    thinking_response = await generate_text(
        client=client,
        prompt=prompt,
        model=model,
        max_tokens=1500,
        temperature=0.3,
        tracer=tracer
    )
    
    thinking = thinking_response.get("text", "").strip()
    
    # Create prompt for final answer
    answer_prompt = f"""
    {prompt}
    {thinking}
    
    Therefore, the answer is:
    """
    
    # Generate final answer
    answer_response = await generate_text(
        client=client,
        prompt=answer_prompt,
        model=model,
        max_tokens=500,
        temperature=0.2,
        tracer=tracer
    )
    
    answer = answer_response.get("text", "").strip()
    
    return {
        "thinking": thinking,
        "answer": answer
    }

async def tree_of_thought(
    client: Any,
    problem: str,
    options_per_step: int = 3,
    max_steps: int = 3,
    model: str = "claude",
    tracer: Optional[LangfuseTracer] = None
) -> Dict[str, Any]:
    """
    Solve a problem using tree-of-thought reasoning.
    
    Args:
        client: Bedrock client
        problem: Problem to solve
        options_per_step: Number of options to consider at each step
        max_steps: Maximum number of steps
        model: Model to use
        tracer: Optional LangfuseTracer instance for monitoring
        
    Returns:
        Dictionary with reasoning tree and solution
    """
    span = None
    if tracer:
        span = tracer.span(
            name="claude_tree_reasoning",
            input={
                "problem_length": len(problem),
                "options_per_step": options_per_step,
                "max_steps": max_steps
            }
        )
    
    try:
        # Import generate_text
        from multiagent.app.tools.bedrock.claude_generate import generate_text
        
        tree = []
        current_step = 0
        current_path = []
        
        while current_step < max_steps:
            # Create prompt for current step
            prompt = f"""
            Solve this problem using a tree of thought approach.
            
            Problem:
            {problem}
            
            Current reasoning path:
            {' -> '.join(current_path) if current_path else 'Starting point'}
            
            Generate {options_per_step} different possible next steps in the reasoning process.
            Label each option clearly as Option 1, Option 2, etc.
            """
            
            # Generate options
            options_response = await generate_text(
                client=client,
                prompt=prompt,
                model=model,
                max_tokens=1000,
                temperature=0.7,
                tracer=tracer
            )
            
            options_text = options_response.get("text", "").strip()
            
            # Extract options using regex
            options = re.findall(r"Option (\d+)[:.]?\s*(.*?)(?=Option \d+|$)", options_text, re.DOTALL)
            if not options:
                options = re.findall(r"(\d+)[).]?\s*(.*?)(?=\d+[).]\s*|$)", options_text, re.DOTALL)
            
            options = [option[1].strip() for option in options]
            
            # Add options to tree
            tree.append({
                "step": current_step,
                "path": current_path.copy(),
                "options": options
            })
            
            # Evaluate options
            eval_prompt = f"""
            Problem:
            {problem}
            
            Current reasoning path:
            {' -> '.join(current_path) if current_path else 'Starting point'}
            
            Options for next step:
            {options_text}
            
            Evaluate each option and select the best one. Explain your reasoning.
            """
            
            eval_response = await generate_text(
                client=client,
                prompt=eval_prompt,
                model=model,
                max_tokens=800,
                temperature=0.2,
                tracer=tracer
            )
            
            eval_text = eval_response.get("text", "").strip()
            
            # Extract best option (most often mentioned in evaluation)
            option_counts = [eval_text.lower().count(f"option {i+1}") for i in range(len(options))]
            best_option_index = option_counts.index(max(option_counts))
            best_option = options[best_option_index]
            
            # Add to current path
            current_path.append(best_option)
            current_step += 1
            
            # Check if we've reached a conclusion
            conclusion_check_prompt = f"""
            Problem:
            {problem}
            
            Current reasoning path:
            {' -> '.join(current_path)}
            
            Have we reached a satisfactory conclusion to the problem? Answer Yes or No.
            """
            
            conclusion_response = await generate_text(
                client=client,
                prompt=conclusion_check_prompt,
                model=model,
                max_tokens=100,
                temperature=0.1,
                tracer=tracer
            )
            
            conclusion_text = conclusion_response.get("text", "").strip().lower()
            if "yes" in conclusion_text:
                break
        
        # Generate final solution
        solution_prompt = f"""
        Problem:
        {problem}
        
        Reasoning path:
        {' -> '.join(current_path)}
        
        Based on this reasoning path, provide the final solution to the problem.
        """
        
        solution_response = await generate_text(
            client=client,
            prompt=solution_prompt,
            model=model,
            max_tokens=500,
            temperature=0.2,
            tracer=tracer
        )
        
        solution = solution_response.get("text", "").strip()
        
        result = {
            "tree": tree,
            "final_path": current_path,
            "solution": solution,
            "steps_taken": current_step
        }
        
        if span:
            span.update(output={"steps_taken": current_step})
        
        return result
        
    except Exception as e:
        logger.error(f"Error in tree-of-thought reasoning: {str(e)}")
        
        if span:
            span.update(output={"error": str(e)})
        
        return {"error": str(e)}

class ClaudeReason:
    """Handles reasoning with Claude."""
    
    def __init__(self):
        """Initialize the Claude reasoner."""
        self.logger = logging.getLogger(__name__)
    
    async def reason(
        self, 
        query: str, 
        context: Optional[str] = None, 
        reasoning_type: str = "step_by_step", 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Performs reasoning with Claude.
        
        Args:
            query: Question or problem
            context: Optional context information
            reasoning_type: Type of reasoning (step_by_step, chain_of_thought, tree_of_thought)
            **kwargs: Additional parameters including:
                - client: Bedrock client (required)
                - model_id: Model ID to use (default: anthropic.claude-3-sonnet-20240229-v1:0)
                - max_tokens: Maximum tokens in response (default varies by reasoning type)
                - temperature: Temperature for generation (default varies by reasoning type)
                - tracer: Optional LangfuseTracer instance
                - options_per_step: For tree_of_thought (default: 3)
                - max_steps: For tree_of_thought (default: 3)
        
        Returns:
            Dict containing reasoning and metadata
        """
        try:
            # Extract parameters
            client = kwargs.get("client")
            if not client:
                raise ValueError("client is required for reasoning")
                
            model_id = kwargs.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
            tracer = kwargs.get("tracer")
            
            span = None
            if tracer:
                span = tracer.span(
                    name=f"claude_{reasoning_type}_reasoning",
                    input={
                        "query_length": len(query),
                        "context_length": len(context) if context else 0,
                        "reasoning_type": reasoning_type,
                        "model_id": model_id
                    }
                )
            
            # Call appropriate reasoning function based on reasoning_type
            if reasoning_type == "step_by_step":
                # Use existing step_by_step_reasoning function
                result = await step_by_step_reasoning(
                    client=client,
                    problem=query,
                    model=model_id,
                    tracer=tracer
                )
                
            elif reasoning_type == "chain_of_thought":
                # Use existing chain_of_thought function
                result = await chain_of_thought(
                    client=client,
                    question=query,
                    context=context,
                    model=model_id,
                    tracer=tracer
                )
                
            elif reasoning_type == "tree_of_thought":
                # Use existing tree_of_thought function
                options_per_step = kwargs.get("options_per_step", 3)
                max_steps = kwargs.get("max_steps", 3)
                
                result = await tree_of_thought(
                    client=client,
                    problem=query,
                    options_per_step=options_per_step,
                    max_steps=max_steps,
                    model=model_id,
                    tracer=tracer
                )
                
            else:
                # Custom reasoning implementation
                from multiagent.app.tools.bedrock.claude_generate import generate_text
                
                # Create custom reasoning prompt
                prompt = self._create_reasoning_prompt(
                    query=query,
                    context=context,
                    reasoning_type=reasoning_type
                )
                
                # Default parameters based on reasoning type
                max_tokens = kwargs.get("max_tokens", 2000)
                temperature = kwargs.get("temperature", 0.2)
                
                # Generate response
                response = await generate_text(
                    client=client,
                    prompt=prompt,
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tracer=tracer
                )
                
                response_text = response.get("text", "").strip()
                
                # Process the response based on reasoning type
                result = self._process_reasoning_response(
                    response_text=response_text,
                    reasoning_type=reasoning_type
                )
            
            # Add metadata to result
            result["reasoning_type"] = reasoning_type
            result["model_id"] = model_id
            
            if span:
                span.update(output={
                    "result_type": type(result).__name__,
                    "success": "error" not in result
                })
            
            return result
            
        except Exception as e:
            error_message = f"Error in Claude reasoning: {str(e)}"
            self.logger.error(error_message)
            
            if span:
                span.update(output={"error": str(e)})
            
            return {
                "error": error_message,
                "reasoning_type": reasoning_type
            }
    
    def _create_reasoning_prompt(
        self, 
        query: str, 
        context: Optional[str] = None, 
        reasoning_type: str = "custom"
    ) -> str:
        """
        Create a prompt for Claude reasoning.
        
        Args:
            query: Question or problem
            context: Optional context information
            reasoning_type: Type of reasoning
            
        Returns:
            Formatted prompt
        """
        # Base prompt with Claude-specific optimizations
        prompt = "Let's break down this problem carefully and systematically.\n\n"
        
        # Add context if provided
        if context:
            prompt += f"""
            Context:
            {context}
            
            """
        
        # Add reasoning type specific instructions
        if reasoning_type == "custom":
            prompt += f"""
            Problem:
            {query}
            
            Please work through this step by step. Consider multiple approaches, evaluate them,
            and arrive at the most robust solution. Explain your reasoning at each stage.
            """
        elif reasoning_type == "mathematical":
            prompt += f"""
            Mathematical Problem:
            {query}
            
            Please solve this mathematical problem step by step. Show all work clearly,
            including any formulas used, calculations performed, and intermediate results.
            Verify your answer by checking if it satisfies all constraints in the problem.
            """
        elif reasoning_type == "causal":
            prompt += f"""
            Causal Analysis Question:
            {query}
            
            Please analyze this question through a causal inference lens. Consider:
            1. The potential causal relationships
            2. Confounding variables that might be present
            3. How to establish causality vs correlation
            4. What evidence would strengthen or weaken causal claims
            """
        elif reasoning_type == "ethical":
            prompt += f"""
            Ethical Analysis Question:
            {query}
            
            Please analyze this question from multiple ethical frameworks:
            1. Consequentialist/utilitarian perspective
            2. Deontological/rights-based perspective
            3. Virtue ethics perspective
            4. Justice and fairness considerations
            
            For each framework, identify key considerations and potential tensions.
            """
        else:
            # Default to general reasoning
            prompt += f"""
            Question:
            {query}
            
            Reasoning process:
            """
        
        return prompt
    
    def _process_reasoning_response(
        self, 
        response_text: str, 
        reasoning_type: str = "custom"
    ) -> Dict[str, Any]:
        """
        Process Claude's reasoning response.
        
        Args:
            response_text: Response from Claude
            reasoning_type: Type of reasoning
            
        Returns:
            Processed reasoning result
        """
        # Process based on reasoning type
        if reasoning_type in ["custom", "mathematical", "causal", "ethical"]:
            # Extract steps if possible
            steps = extract_reasoning_steps(response_text)
            
            # Extract conclusion if present
            conclusion_patterns = [
                r"(?:In conclusion|Therefore|Thus|To summarize|Finally)(.*?)$",
                r"(?:The answer is|The solution is|The result is)(.*?)$"
            ]
            
            conclusion = ""
            for pattern in conclusion_patterns:
                matches = re.findall(pattern, response_text, re.DOTALL)
                if matches:
                    conclusion = matches[0].strip()
                    break
            
            return {
                "steps": steps,
                "conclusion": conclusion,
                "full_reasoning": response_text,
                "step_count": len(steps)
            }
        else:
            # Default processing
            return {
                "reasoning": response_text
            }