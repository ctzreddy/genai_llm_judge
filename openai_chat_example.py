import os
from dotenv import load_dotenv
from openai import OpenAI
from llm_judge import LLMJudge, JudgeType

# Get API key from environment variable
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Initialize the LLM Judge
judge = LLMJudge(judge_model="gpt-4o", temperature=0.3)


def chat_with_llm(prompt, model="gpt-4o", max_tokens=150, temperature=0.7):
    """
    Sends a prompt to the OpenAI LLM and returns the response.
    
    Args:
        prompt: The user prompt
        model: Model to use (default: gpt-4o)
        max_tokens: Maximum tokens in response
        temperature: Temperature setting.
        
    Returns:
        The LLM response string or None on error
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None


def chat_with_judge(prompt, judge_type=JudgeType.QUALITY, passing_score=70):
    """
    Get LLM response and have it judged by an LLM judge.
    
    Args:
        prompt: The user prompt
        judge_type: Type of judgment to perform
        passing_score: Minimum score to pass
        
    Returns:
        Dictionary with response and judgment results
    """
    # Get response from LLM
    response = chat_with_llm(prompt)
    
    if not response:
        return {
            "response": None,
            "judgment": None,
            "error": "Failed to get response from LLM"
        }
    
    # Judge the response
    judgment = judge.judge(
        original_prompt=prompt,
        response=response,
        judge_type=judge_type,
        passing_score=passing_score
    )
    
    return {
        "response": response,
        "judgment": judgment
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Basic LLM Response with Quality Judgment")
    print("=" * 70)
    
    user_prompt = "Tell me a short, interesting fact about space."
    result = chat_with_judge(user_prompt, judge_type=JudgeType.QUALITY)
    
    if result["response"]:
        print(f"\nUser Prompt: {user_prompt}")
        print(f"\nLLM Response: {result['response']}")
        
        if result["judgment"] and "error" not in result["judgment"]:
            judgment = result["judgment"]
            print(f"\n{'='*70}")
            print("JUDGMENT RESULTS")
            print(f"{'='*70}")
            print(f"Judge Type: {judgment.get('judge_type', 'N/A')}")
            print(f"Score: {judgment.get('score', 0)}/100")
            print(f"Passed: {'✓ YES' if judgment.get('passed') else '✗ NO'}")
            print(f"\nFeedback: {judgment.get('feedback', 'N/A')}")
            
            # Show strengths/weaknesses if available
            judgment_data = judgment.get('judgment', {})
            if 'strengths' in judgment_data:
                print(f"\nStrengths:")
                for strength in judgment_data['strengths']:
                    print(f"  • {strength}")
            if 'weaknesses' in judgment_data:
                print(f"\nWeaknesses:")
                for weakness in judgment_data['weaknesses']:
                    print(f"  • {weakness}")
    
    print("\n" + "=" * 70)
    print("Example 2: Correctness Judgment")
    print("=" * 70)
    
    factual_prompt = "What is the capital of France?"
    result2 = chat_with_judge(factual_prompt, judge_type=JudgeType.CORRECTNESS)
    
    if result2["response"]:
        print(f"\nUser Prompt: {factual_prompt}")
        print(f"LLM Response: {result2['response']}")
        
        if result2["judgment"] and "error" not in result2["judgment"]:
            judgment = result2["judgment"]
            print(f"\nCorrectness Score: {judgment.get('score', 0)}/100")
            print(f"Passed: {'✓ YES' if judgment.get('passed') else '✗ NO'}")
            print(f"Feedback: {judgment.get('feedback', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("Example 3: Comparing Two Responses")
    print("=" * 70)
    
    comparison_prompt = "Explain what Python is in one sentence."
    response1 = chat_with_llm(comparison_prompt, max_tokens=50)
    response2 = chat_with_llm(comparison_prompt, max_tokens=50, temperature=0.9)
    
    if response1 and response2:
        print(f"\nPrompt: {comparison_prompt}")
        print(f"\nResponse 1: {response1}")
        print(f"Response 2: {response2}")
        
        comparison = judge.compare_responses(
            original_prompt=comparison_prompt,
            response1=response1,
            response2=response2
        )
        
        if "error" not in comparison:
            comp_data = comparison.get("comparison", {})
            print(f"\n{'='*70}")
            print("COMPARISON RESULTS")
            print(f"{'='*70}")
            print(f"Winner: Response {comparison.get('winner', 'N/A')}")
            print(f"Response 1 Score: {comparison.get('response1_score', 0)}/100")
            print(f"Response 2 Score: {comparison.get('response2_score', 0)}/100")
            print(f"\nWinner Explanation: {comp_data.get('winner_explanation', 'N/A')}")