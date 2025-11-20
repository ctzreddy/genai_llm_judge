"""
Examples demonstrating different validation patterns for LLM responses.
"""

from typing import Tuple, Optional
from llm_validator import (
    ResponseValidator,
    NotEmptyValidator,
    LengthValidator,
    JSONValidator,
    JSONSchemaValidator,
    ContainsValidator,
    RegexValidator,
    CustomValidator,
    ValidationError
)


def example_basic_validation():
    """Example: Basic validation - not empty and length check."""
    print("\n=== Basic Validation ===")
    
    validator = ResponseValidator([
        NotEmptyValidator(),
        LengthValidator(min_length=10, max_length=200)
    ])
    
    # Valid response
    result = validator.validate("This is a valid response that meets the criteria.")
    print(f"Valid response: {result['is_valid']}")
    
    # Invalid - too short
    result = validator.validate("Short")
    print(f"Short response: {result['is_valid']}, Errors: {result['errors']}")
    
    # Invalid - empty
    result = validator.validate("")
    print(f"Empty response: {result['is_valid']}, Errors: {result['errors']}")


def example_json_validation():
    """Example: JSON format validation."""
    print("\n=== JSON Validation ===")
    
    validator = ResponseValidator([
        NotEmptyValidator(),
        JSONValidator()
    ])
    
    # Valid JSON
    result = validator.validate('{"name": "John", "age": 30}')
    print(f"Valid JSON: {result['is_valid']}")
    
    # Invalid JSON
    result = validator.validate('{"name": "John", "age": 30')  # Missing closing brace
    print(f"Invalid JSON: {result['is_valid']}, Errors: {result['errors']}")


def example_json_schema_validation():
    """Example: JSON schema validation with required keys and types."""
    print("\n=== JSON Schema Validation ===")
    
    validator = ResponseValidator([
        NotEmptyValidator(),
        JSONValidator(),
        JSONSchemaValidator(
            required_keys=['name', 'age', 'email'],
            key_types={'name': str, 'age': int, 'email': str}
        )
    ])
    
    # Valid JSON with correct schema
    result = validator.validate('{"name": "John", "age": 30, "email": "john@example.com"}')
    print(f"Valid schema: {result['is_valid']}")
    
    # Invalid - missing required key
    result = validator.validate('{"name": "John", "age": 30}')
    print(f"Missing key: {result['is_valid']}, Errors: {result['errors']}")
    
    # Invalid - wrong type
    result = validator.validate('{"name": "John", "age": "thirty", "email": "john@example.com"}')
    print(f"Wrong type: {result['is_valid']}, Errors: {result['errors']}")


def example_content_validation():
    """Example: Content validation - checking for keywords."""
    print("\n=== Content Validation ===")
    
    validator = ResponseValidator([
        NotEmptyValidator(),
        ContainsValidator(
            keywords=['Python', 'programming', 'code'],
            case_sensitive=False,
            all_required=False  # At least one keyword must be present
        )
    ])
    
    # Valid - contains keyword
    result = validator.validate("Python is a great programming language.")
    print(f"Contains keyword: {result['is_valid']}")
    
    # Invalid - no keywords
    result = validator.validate("This is about something else entirely.")
    print(f"No keywords: {result['is_valid']}, Errors: {result['errors']}")


def example_regex_validation():
    """Example: Regex pattern validation."""
    print("\n=== Regex Validation ===")
    
    # Validate email format
    validator = ResponseValidator([
        NotEmptyValidator(),
        RegexValidator(
            pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$',
            flags=0
        )
    ])
    
    # Valid email
    result = validator.validate("user@example.com")
    print(f"Valid email: {result['is_valid']}")
    
    # Invalid email
    result = validator.validate("not-an-email")
    print(f"Invalid email: {result['is_valid']}, Errors: {result['errors']}")


def example_custom_validation():
    """Example: Custom validation function."""
    print("\n=== Custom Validation ===")
    
    def validate_no_profanity(response: str) -> Tuple[bool, Optional[str]]:
        """Custom validator to check for profanity."""
        bad_words = ['badword1', 'badword2']  # Example list
        response_lower = response.lower()
        
        for word in bad_words:
            if word in response_lower:
                return False, f"Response contains inappropriate content: {word}"
        
        return True, None
    
    validator = ResponseValidator([
        NotEmptyValidator(),
        CustomValidator(validate_no_profanity)
    ])
    
    # Valid - no profanity
    result = validator.validate("This is a clean response.")
    print(f"Clean response: {result['is_valid']}")
    
    # Invalid - contains profanity (if bad_words were in the response)
    # This is just an example - actual implementation would check real words


def example_combined_validation():
    """Example: Combining multiple validators."""
    print("\n=== Combined Validation ===")
    
    # Example: Validate a JSON response with specific schema
    validator = ResponseValidator([
        NotEmptyValidator(),
        LengthValidator(min_length=20, max_length=1000),
        JSONValidator(),
        JSONSchemaValidator(
            required_keys=['status', 'message'],
            key_types={'status': str, 'message': str}
        ),
        ContainsValidator(keywords=['success', 'error'], all_required=False)
    ])
    
    # Valid response
    result = validator.validate('{"status": "success", "message": "Operation completed successfully"}')
    print(f"Valid combined: {result['is_valid']}")
    
    # Invalid - missing required key
    result = validator.validate('{"status": "success"}')
    print(f"Missing key: {result['is_valid']}, Errors: {result['errors']}")


if __name__ == "__main__":
    print("LLM Response Validation Framework - Examples")
    print("=" * 60)
    
    example_basic_validation()
    example_json_validation()
    example_json_schema_validation()
    example_content_validation()
    example_regex_validation()
    example_custom_validation()
    example_combined_validation()
    
    print("\n" + "=" * 60)
    print("All examples completed!")

