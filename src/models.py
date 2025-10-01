from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

# Common Models
class Model(BaseModel):
    id: str
    object: str = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = "google"

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]


# Gemini Models
class GeminiPart(BaseModel):
    text: Optional[str] = None
    inlineData: Optional[Dict[str, Any]] = None
    fileData: Optional[Dict[str, Any]] = None
    thought: Optional[bool] = False

class GeminiContent(BaseModel):
    role: str
    parts: List[GeminiPart]

class GeminiSystemInstruction(BaseModel):
    parts: List[GeminiPart]

class GeminiGenerationConfig(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    topP: Optional[float] = Field(None, ge=0.0, le=1.0)
    topK: Optional[int] = Field(None, ge=1)
    maxOutputTokens: Optional[int] = Field(None, ge=1)
    stopSequences: Optional[List[str]] = None
    responseMimeType: Optional[str] = None
    responseSchema: Optional[Dict[str, Any]] = None
    candidateCount: Optional[int] = Field(None, ge=1, le=8)
    seed: Optional[int] = None
    frequencyPenalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presencePenalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    thinkingConfig: Optional[Dict[str, Any]] = None

class GeminiSafetySetting(BaseModel):
    category: str
    threshold: str

class GeminiRequest(BaseModel):
    contents: List[GeminiContent]
    systemInstruction: Optional[GeminiSystemInstruction] = None
    generationConfig: Optional[GeminiGenerationConfig] = None
    safetySettings: Optional[List[GeminiSafetySetting]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    toolConfig: Optional[Dict[str, Any]] = None
    cachedContent: Optional[str] = None
    
    class Config:
        extra = "allow"  # 允许透传未定义的字段

class GeminiCandidate(BaseModel):
    content: GeminiContent
    finishReason: Optional[str] = None
    index: int = 0
    safetyRatings: Optional[List[Dict[str, Any]]] = None
    citationMetadata: Optional[Dict[str, Any]] = None
    tokenCount: Optional[int] = None

class GeminiUsageMetadata(BaseModel):
    promptTokenCount: Optional[int] = None
    candidatesTokenCount: Optional[int] = None
    totalTokenCount: Optional[int] = None

class GeminiResponse(BaseModel):
    candidates: List[GeminiCandidate]
    usageMetadata: Optional[GeminiUsageMetadata] = None
    modelVersion: Optional[str] = None

# Error Models
class APIError(BaseModel):
    message: str
    type: str = "api_error"
    code: Optional[int] = None

class ErrorResponse(BaseModel):
    error: APIError

# Control Panel Models
class SystemStatus(BaseModel):
    status: str
    timestamp: str
    credentials: Dict[str, int]
    config: Dict[str, Any]
    current_credential: str

class CredentialInfo(BaseModel):
    filename: str
    project_id: Optional[str] = None
    status: Dict[str, Any]
    size: Optional[int] = None
    modified_time: Optional[str] = None
    error: Optional[str] = None

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    module: Optional[str] = None

class ConfigValue(BaseModel):
    key: str
    value: Any
    env_locked: bool = False
    description: Optional[str] = None

# Authentication Models
class AuthRequest(BaseModel):
    project_id: Optional[str] = None
    user_session: Optional[str] = None

class AuthResponse(BaseModel):
    success: bool
    auth_url: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    requires_manual_project_id: Optional[bool] = None
    requires_project_selection: Optional[bool] = None
    available_projects: Optional[List[Dict[str, str]]] = None

class CredentialStatus(BaseModel):
    disabled: bool = False
    error_codes: List[int] = []
    last_success: Optional[str] = None
