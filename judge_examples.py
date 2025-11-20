"""
Comprehensive examples demonstrating the LLM-as-a-Judge framework.
"""

from llm_judge import LLMJudge, JudgeType


def example_quality_judgment():
    """Example: Quality judgment of an LLM response."""
    print("\n" + "=" * 70)
    print("Example: Quality Judgment")
    print("=" * 70)
    
    judge = LLMJudge(judge_model="gpt-4o", temperature=0.3)
    
    prompt = "Explain quantum computing in simple terms."
    response = "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously, allowing for parallel processing and potentially solving certain problems much faster than classical computers."
    
    result = judge.judge(
        original_prompt=prompt,
        response=response,
        judge_type=JudgeType.QUALITY,
        passing_score=70
    )
    
    print(f"Prompt: {prompt}")
    print(f"\nResponse: {response}")
    print(f"\n{'='*70}")
    print("JUDGMENT RESULTS")
    print(f"{'='*70}")
    print(f"Score: {result.get('score', 0)}/100")
    print(f"Passed: {'✓ YES' if result.get('passed') else '✗ NO'}")
    print(f"\nFeedback:\n{result.get('feedback', 'N/A')}")
    
    judgment_data = result.get('judgment', {})
    if 'strengths' in judgment_data:
        print(f"\nStrengths:")
        for strength in judgment_data['strengths']:
            print(f"  • {strength}")
    if 'weaknesses' in judgment_data:
        print(f"\nWeaknesses:")
        for weakness in judgment_data['weaknesses']:
            print(f"  • {weakness}")


def example_correctness_judgment():
    """Example: Correctness judgment of a factual response."""
    print("\n" + "=" * 70)
    print("Example: Correctness Judgment")
    print("=" * 70)
    
    judge = LLMJudge(judge_model="gpt-4o", temperature=0.3)
    
    prompt = "What is the speed of light in vacuum?"
    response = "The speed of light in vacuum is approximately 299,792,458 meters per second, or about 3 × 10^8 m/s."
    
    result = judge.judge(
        original_prompt=prompt,
        response=response,
        judge_type=JudgeType.CORRECTNESS,
        passing_score=80
    )
    
    print(f"Prompt: {prompt}")
    print(f"\nResponse: {response}")
    print(f"\n{'='*70}")
    print("JUDGMENT RESULTS")
    print(f"{'='*70}")
    print(f"Score: {result.get('score', 0)}/100")
    print(f"Passed: {'✓ YES' if result.get('passed') else '✗ NO'}")
    print(f"\nFeedback:\n{result.get('feedback', 'N/A')}")
    
    judgment_data = result.get('judgment', {})
    if 'errors_found' in judgment_data:
        print(f"\nErrors Found:")
        for error in judgment_data['errors_found']:
            print(f"  • {error}")
    if 'correct_aspects' in judgment_data:
        print(f"\nCorrect Aspects:")
        for aspect in judgment_data['correct_aspects']:
            print(f"  • {aspect}")


def example_appropriateness_judgment():
    """Example: Appropriateness judgment of a response."""
    print("\n" + "=" * 70)
    print("Example: Appropriateness Judgment")
    print("=" * 70)
    
    judge = LLMJudge(judge_model="gpt-4o", temperature=0.3)
    
    prompt = "How should I handle a difficult conversation with my manager?"
    response = "When approaching a difficult conversation with your manager, it's important to prepare your points, choose an appropriate time and setting, remain professional and respectful, listen actively, and focus on finding solutions rather than placing blame."
    
    result = judge.judge(
        original_prompt=prompt,
        response=response,
        judge_type=JudgeType.APPROPRIATENESS,
        passing_score=70
    )
    
    print(f"Prompt: {prompt}")
    print(f"\nResponse: {response}")
    print(f"\n{'='*70}")
    print("JUDGMENT RESULTS")
    print(f"{'='*70}")
    print(f"Score: {result.get('score', 0)}/100")
    print(f"Passed: {'✓ YES' if result.get('passed') else '✗ NO'}")
    print(f"\nFeedback:\n{result.get('feedback', 'N/A')}")
    
    judgment_data = result.get('judgment', {})
    if 'concerns' in judgment_data:
        print(f"\nConcerns:")
        for concern in judgment_data['concerns']:
            print(f"  • {concern}")
    if 'appropriate_aspects' in judgment_data:
        print(f"\nAppropriate Aspects:")
        for aspect in judgment_data['appropriate_aspects']:
            print(f"  • {aspect}")


def example_comprehensiveness_judgment():
    """Example: Comprehensiveness judgment of a response."""
    print("\n" + "=" * 70)
    print("Example: Comprehensiveness Judgment")
    print("=" * 70)
    
    judge = LLMJudge(judge_model="gpt-4o", temperature=0.3)
    
    prompt = "What are the main benefits of exercise?"
    response = "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and increased energy levels."
    
    result = judge.judge(
        original_prompt=prompt,
        response=response,
        judge_type=JudgeType.COMPREHENSIVENESS,
        passing_score=70
    )
    
    print(f"Prompt: {prompt}")
    print(f"\nResponse: {response}")
    print(f"\n{'='*70}")
    print("JUDGMENT RESULTS")
    print(f"{'='*70}")
    print(f"Score: {result.get('score', 0)}/100")
    print(f"Passed: {'✓ YES' if result.get('passed') else '✗ NO'}")
    print(f"\nFeedback:\n{result.get('feedback', 'N/A')}")
    
    judgment_data = result.get('judgment', {})
    if 'covered_aspects' in judgment_data:
        print(f"\nCovered Aspects:")
        for aspect in judgment_data['covered_aspects']:
            print(f"  • {aspect}")
    if 'missing_aspects' in judgment_data:
        print(f"\nMissing Aspects:")
        for aspect in judgment_data['missing_aspects']:
            print(f"  • {aspect}")


def example_custom_judgment():
    """Example: Custom judgment with specific criteria."""
    print("\n" + "=" * 70)
    print("Example: Custom Judgment")
    print("=" * 70)
    
    judge = LLMJudge(judge_model="gpt-4o", temperature=0.3)
    
    prompt = "Write a product description for a wireless mouse."
    response = "Introducing our premium wireless mouse - featuring ergonomic design, precision tracking, long battery life, and seamless connectivity. Perfect for professionals and gamers alike."
    
    custom_criteria = """Evaluate this product description based on:
- Marketing effectiveness and persuasiveness
- Clarity of key features
- Appeal to target audience
- Professional tone
- Completeness of information"""
    
    result = judge.judge(
        original_prompt=prompt,
        response=response,
        judge_type=JudgeType.CUSTOM,
        criteria=custom_criteria,
        passing_score=75
    )
    
    print(f"Prompt: {prompt}")
    print(f"\nResponse: {response}")
    print(f"\n{'='*70}")
    print("JUDGMENT RESULTS")
    print(f"{'='*70}")
    print(f"Score: {result.get('score', 0)}/100")
    print(f"Passed: {'✓ YES' if result.get('passed') else '✗ NO'}")
    print(f"\nFeedback:\n{result.get('feedback', 'N/A')}")


def example_compare_responses():
    """Example: Comparing two different responses."""
    print("\n" + "=" * 70)
    print("Example: Comparing Two Responses")
    print("=" * 70)
    
    judge = LLMJudge(judge_model="gpt-4o", temperature=0.3)
    
    prompt = "What is machine learning?"
    
    response1 = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
    
    response2 = "Machine learning is a method of data analysis that automates analytical model building. It's a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention. It includes various techniques like supervised learning, unsupervised learning, and reinforcement learning."
    
    print(f"Prompt: {prompt}")
    print(f"\nResponse 1: {response1}")
    print(f"\nResponse 2: {response2}")
    
    comparison = judge.compare_responses(
        original_prompt=prompt,
        response1=response1,
        response2=response2
    )
    
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    
    if "error" not in comparison:
        comp_data = comparison.get("comparison", {})
        print(f"Winner: Response {comparison.get('winner', 'N/A')}")
        print(f"Response 1 Score: {comparison.get('response1_score', 0)}/100")
        print(f"Response 2 Score: {comparison.get('response2_score', 0)}/100")
        print(f"\nWinner Explanation:\n{comp_data.get('winner_explanation', 'N/A')}")
        
        if 'response1_strengths' in comp_data:
            print(f"\nResponse 1 Strengths:")
            for strength in comp_data['response1_strengths']:
                print(f"  • {strength}")
        
        if 'response2_strengths' in comp_data:
            print(f"\nResponse 2 Strengths:")
            for strength in comp_data['response2_strengths']:
                print(f"  • {strength}")
    else:
        print(f"Error: {comparison.get('error', 'Unknown error')}")


def example_judge_multiple():
    """Example: Judging multiple responses at once."""
    print("\n" + "=" * 70)
    print("Example: Judging Multiple Responses")
    print("=" * 70)
    
    judge = LLMJudge(judge_model="gpt-4o", temperature=0.3)
    
    prompt = "What is Python?"
    
    responses = [
        "Python is a programming language.",
        "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",
        "Python is a snake."
    ]
    
    print(f"Prompt: {prompt}\n")
    
    results = judge.judge_multiple(
        original_prompt=prompt,
        responses=responses,
        judge_type=JudgeType.QUALITY,
        passing_score=60
    )
    
    for i, (response, result) in enumerate(zip(responses, results), 1):
        print(f"{'='*70}")
        print(f"Response {i}: {response}")
        print(f"{'='*70}")
        print(f"Score: {result.get('score', 0)}/100")
        print(f"Passed: {'✓ YES' if result.get('passed') else '✗ NO'}")
        print(f"Feedback: {result.get('feedback', 'N/A')[:200]}...")
        print()


if __name__ == "__main__":
    print("LLM-as-a-Judge Framework - Comprehensive Examples")
    print("=" * 70)
    
    example_quality_judgment()
    example_correctness_judgment()
    example_appropriateness_judgment()
    example_comprehensiveness_judgment()
    example_custom_judgment()
    example_compare_responses()
    example_judge_multiple()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)

