# tests/test_models.py

from typing import List, Dict, Any
from pydantic import BaseModel, Field

class CompletionMetadata(BaseModel):
    """Model for completion metadata"""
    model: str = Field(description="Model identifier")
    tokens_used: int = Field(description="Token count")
    processing_time: float = Field(description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "required": ["model", "tokens_used", "processing_time"],
            "additionalProperties": False
        }

class StructuredOutput(BaseModel):
    """Standard structured output"""
    text: str = Field(description="Generated text content")
    metadata: CompletionMetadata
    
    class Config:
        json_schema_extra = {
            "required": ["text", "metadata"],
            "additionalProperties": False
        }

class BatchMetadata(BaseModel):
    """Batch processing metadata"""
    batch_id: str = Field(description="Unique batch identifier")
    model: str = Field(description="Model used")
    total_prompts: int = Field(description="Total number of prompts")
    successful_prompts: int = Field(description="Number of successful prompts")
    failed_prompts: int = Field(description="Number of failed prompts")
    processing_time: float = Field(description="Total processing time")
    errors: List[Dict[str, str]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "required": ["total_prompts", "successful_prompts", "failed_prompts"],
            "additionalProperties": False
        }

class BatchResult(BaseModel):
    """Batch processing results"""
    metadata: BatchMetadata
    results: List[Dict[str, Any]] = Field(description="List of results")
    
    class Config:
        json_schema_extra = {
            "required": ["metadata", "results"],
            "additionalProperties": False
        }
