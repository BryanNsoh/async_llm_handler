import os
from typing import Any, List, Optional, Type, TypeVar, Union, Literal
from pydantic import BaseModel, Field, field_validator
import google.generativeai as genai
import instructor
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

VALID_MODELS = Literal[
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-8b"
]

class GeminiHandler:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        logger.info("GeminiHandler initialized successfully")

    def _validate_model(self, model: str) -> None:
        valid_models = ["gemini-1.5-pro-002", "gemini-1.5-pro-002", "gemini-1.5-flash-8b"]
        if model not in valid_models:
            raise ValueError(f"Invalid model. Must be one of: {', '.join(valid_models)}")

    async def process(
        self,
        prompts: Union[str, List[str]],
        model: VALID_MODELS = "gemini-1.5-flash-8b",
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        response_format: Optional[Type[T]] = None,
        max_retries: int = 3,
        output_dir: Optional[str] = None,
    ) -> Union[Any, T, List[T]]:
        """
        Process prompts using Gemini models with Instructor for structured output.
        
        Args:
            prompts: Single prompt or list of prompts
            model: Gemini model version to use
            system_message: Optional system message for context
            temperature: Generation temperature (0.0 to 1.0)
            response_format: Pydantic model for structured output (required)
            max_retries: Maximum number of retries for validation failures
            output_dir: Optional directory for saving results
            
        Returns:
            Processed results in specified format
        """
        self._validate_model(model)
        logger.info(f"Processing with model: {model}")
        
        # Convert single prompt to list for unified processing
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        
        try:
            # Initialize Gemini model
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config={"temperature": temperature}
            )
            
            # Patch with Instructor
            client = instructor.from_gemini(
                client=gemini_model,
                mode=instructor.Mode.GEMINI_JSON,
            )
            
            results = []
            total_prompts = len(prompt_list)
            
            for idx, prompt in enumerate(prompt_list, 1):
                try:
                    messages = []
                    if system_message:
                        messages.append({"role": "system", "content": system_message})
                    messages.append({"role": "user", "content": prompt})
                    
                    logger.info(f"Processing prompt {idx}/{total_prompts}")
                    
                    response = client.create(
                        messages=messages,
                        response_model=response_format,
                        max_retries=max_retries
                    )
                    
                    results.append(response)
                    
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"result_{timestamp}_{idx}.json"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'w') as f:
                            f.write(response.model_dump_json(indent=2))
                    
                except Exception as e:
                    logger.error(f"Error processing prompt {idx}: {str(e)}")
                    results.append(None)
            
            if isinstance(prompts, str):
                return results[0] if results else None
                
            return results
            
        except Exception as e:
            logger.error(f"Critical error in process: {str(e)}")
            raise

# Example usage demonstrating Instructor's capabilities
async def main():
    # Define Pydantic models with V2 validators
    class UserInfo(BaseModel):
        """Extract user information with validation."""
        name: str = Field(description="The person's full name")
        age: int = Field(description="The person's age in years")
        occupation: str = Field(description="The person's job or profession")
        
        @field_validator('age')
        @classmethod
        def validate_age(cls, v: int) -> int:
            if v < 0 or v > 120:
                raise ValueError("Age must be between 0 and 120")
            return v

    class MovieReview(BaseModel):
        """Structured movie review with rating validation."""
        title: str = Field(description="The movie title")
        rating: float = Field(description="Rating out of 10")
        summary: str = Field(description="Brief review summary")
        genres: List[str] = Field(description="List of movie genres")
        
        @field_validator('rating')
        @classmethod
        def validate_rating(cls, v: float) -> float:
            if v < 0 or v > 10:
                raise ValueError("Rating must be between 0 and 10")
            return v

    handler = GeminiHandler()
    
    # Test all models with both UserInfo and MovieReview
    models = [
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash-8b"
    ]
    
    test_cases = [
        {
            "model_type": UserInfo,
            "prompt": "Extract information about John who is a 30 year old software engineer",
            "description": "User Information Extraction"
        },
        {
            "model_type": MovieReview,
            "prompt": "Review the movie 'Inception': A mind-bending thriller about dreams within dreams. Amazing special effects. 9/10",
            "description": "Movie Review Processing"
        }
    ]
    
    # Run tests for each combination of model and test case
    for model in models:
        logger.info(f"\n=== Testing Model: {model} ===")
        
        for test_case in test_cases:
            try:
                logger.info(f"\nRunning {test_case['description']}")
                result = await handler.process(
                    prompts=test_case["prompt"],
                    model=model,
                    temperature=0.7,
                    response_format=test_case["model_type"],
                    max_retries=2,
                    output_dir=f"test_outputs_{model.replace('-', '_')}"
                )
                logger.info(f"Result: {result}")
                
            except Exception as e:
                logger.error(f"Error testing {model} with {test_case['description']}: {str(e)}")
                continue
        
        # Add a batch test for each model
        try:
            logger.info(f"\nRunning Batch Test for {model}")
            batch_prompts = [
                "Extract information about Sarah who is a 25 year old teacher",
                "Extract information about Mike who is a 45 year old doctor",
                "Extract information about Emma who is a 28 year old artist"
            ]
            
            batch_results = await handler.process(
                prompts=batch_prompts,
                model=model,
                temperature=0.7,
                response_format=UserInfo,
                max_retries=2,
                output_dir=f"batch_outputs_{model.replace('-', '_')}"
            )
            
            logger.info("\nBatch processing results:")
            for idx, result in enumerate(batch_results, 1):
                logger.info(f"Batch Result {idx}: {result}")
                
        except Exception as e:
            logger.error(f"Error in batch test for {model}: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())