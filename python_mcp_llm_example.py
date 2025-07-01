#!/usr/bin/env python3
"""
Python MCP + LLM Implementation
Roo-Code의 MCP 구현을 Python으로 변환한 예제

주요 기능:
1. MCP 서버 연결 및 관리
2. 시스템 프롬프트에 MCP 정보 동적 포함
3. HTTP POST로 LLM API 호출
4. MCP 도구 실행 및 결과 처리
"""

import asyncio
import json
import subprocess
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
import aiofiles
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ============================================================================
# 1. 데이터 모델 정의
# ============================================================================

class McpServerType(Enum):
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"

@dataclass
class McpServerConfig:
    name: str
    type: McpServerType
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: int = 60
    always_allow: List[str] = None
    disabled_tools: List[str] = None

    def __post_init__(self):
        if self.always_allow is None:
            self.always_allow = []
        if self.disabled_tools is None:
            self.disabled_tools = []

@dataclass
class McpTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str

@dataclass
class McpResource:
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None

@dataclass
class LLMMessage:
    role: str  # "user", "assistant", "system"
    content: str

@dataclass
class LLMConfig:
    provider: str  # "openai", "anthropic", "ollama"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000

# ============================================================================
# 2. MCP Hub - MCP 서버 관리 클래스
# ============================================================================

class McpHub:
    """MCP 서버들을 연결하고 관리하는 핵심 클래스"""
    
    def __init__(self):
        self.connections: Dict[str, ClientSession] = {}
        self.servers: Dict[str, McpServerConfig] = {}
        self.tools: Dict[str, List[McpTool]] = {}
        self.resources: Dict[str, List[McpResource]] = {}
        
    async def connect_server(self, config: McpServerConfig) -> bool:
        """MCP 서버에 연결"""
        try:
            if config.type == McpServerType.STDIO:
                print(f"🔗 MCP 서버 '{config.name}' 연결 시도 중...")
                
                # Stdio 기반 MCP 서버 연결
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                    env=config.env or {}
                )
                
                # 검증된 수동 MCP 연결 방식 사용
                try:
                    # 1. 서버 프로세스 시작
                    import subprocess
                    process = subprocess.Popen(
                        [config.command] + (config.args or []),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',  # UTF-8 인코딩 명시적 설정
                        bufsize=0
                    )
                    
                    # 2. 초기화 요청
                    init_request = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "roots": {"listChanged": True},
                                "sampling": {}
                            },
                            "clientInfo": {
                                "name": "python-mcp-llm-client",
                                "version": "1.0.0"
                            }
                        }
                    }
                    
                    # 요청 전송
                    process.stdin.write(json.dumps(init_request) + "\n")
                    process.stdin.flush()
                    
                    # 응답 대기
                    response_line = await asyncio.wait_for(
                        asyncio.to_thread(process.stdout.readline), 
                        timeout=10
                    )
                    response = json.loads(response_line.strip())
                    
                    if "error" in response:
                        raise Exception(f"초기화 실패: {response['error']}")
                    
                    # 3. initialized 알림 전송
                    init_notification = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    }
                    process.stdin.write(json.dumps(init_notification) + "\n")
                    process.stdin.flush()
                    
                    # 4. 연결 정보 저장
                    self.connections[config.name] = process
                    self.servers[config.name] = config
                    
                    print(f"✅ MCP 서버 '{config.name}' 연결 성공")
                    
                except asyncio.TimeoutError:
                    print(f"❌ MCP 서버 '{config.name}' 초기화 타임아웃")
                    return False
                except Exception as init_error:
                    print(f"❌ MCP 서버 '{config.name}' 초기화 실패: {init_error}")
                    return False
                
                # 서버의 도구와 리소스 목록 가져오기
                await self._fetch_server_capabilities(config.name)
                
                print(f"✅ MCP 서버 '{config.name}' 연결 성공")
                return True
                
            else:
                # SSE, HTTP 연결은 추후 구현
                print(f"❌ {config.type.value} 타입은 아직 구현되지 않음")
                return False
                
        except Exception as e:
            print(f"❌ MCP 서버 '{config.name}' 연결 실패: {e}")
            return False
    
    async def _fetch_server_capabilities(self, server_name: str):
        """서버의 도구와 리소스 목록 가져오기"""
        process = self.connections[server_name]
        
        # 도구 목록 가져오기
        try:
            # JSON-RPC 요청으로 직접 통신
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            process.stdin.write(json.dumps(tools_request) + "\n")
            process.stdin.flush()
            
            response_line = await asyncio.wait_for(
                asyncio.to_thread(process.stdout.readline), 
                timeout=5
            )
            tools_response = json.loads(response_line.strip())
            
            if "error" not in tools_response:
                tools = tools_response["result"]["tools"]
                self.tools[server_name] = [
                    McpTool(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        input_schema=tool.get("inputSchema", {}),
                        server_name=server_name
                    )
                    for tool in tools
                ]
                print(f"📚 서버 '{server_name}' 도구 {len(tools)}개 발견")
            else:
                print(f"⚠️  서버 '{server_name}' 도구 목록 가져오기 실패: {tools_response['error']}")
                self.tools[server_name] = []
        except Exception as e:
            print(f"⚠️  서버 '{server_name}' 도구 목록 가져오기 실패: {e}")
            self.tools[server_name] = []
        
        # 리소스 목록 가져오기 (간단화)
        self.resources[server_name] = []  # 일단 스킵
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP 도구 실행"""
        if server_name not in self.connections:
            raise ValueError(f"MCP 서버 '{server_name}'이 연결되지 않음")
        
        process = self.connections[server_name]
        
        try:
            # JSON-RPC 도구 실행 요청
            call_request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            process.stdin.write(json.dumps(call_request) + "\n")
            process.stdin.flush()
            
            response_line = await asyncio.wait_for(
                asyncio.to_thread(process.stdout.readline), 
                timeout=30
            )
            call_response = json.loads(response_line.strip())
            
            if "error" in call_response:
                return {
                    "success": False,
                    "content": f"도구 실행 오류: {call_response['error']}",
                    "is_error": True
                }
            
            result = call_response["result"]
            
            # 결과 처리
            content_parts = []
            for content in result.get("content", []):
                if content.get("type") == "text":
                    content_parts.append(content.get("text", ""))
            
            return {
                "success": True,
                "content": "\n".join(content_parts),
                "is_error": result.get("isError", False)
            }
            
        except Exception as e:
            return {
                "success": False,
                "content": f"도구 실행 오류: {str(e)}",
                "is_error": True
            }
    
    async def read_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        """MCP 리소스 읽기"""
        if server_name not in self.connections:
            raise ValueError(f"MCP 서버 '{server_name}'이 연결되지 않음")
        
        session = self.connections[server_name]
        
        try:
            result = await session.read_resource(uri)
            
            content_parts = []
            for content in result.contents:
                if hasattr(content, 'text'):
                    content_parts.append(content.text)
            
            return {
                "success": True,
                "content": "\n".join(content_parts),
                "mime_type": getattr(result.contents[0], 'mimeType', None) if result.contents else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "content": f"리소스 읽기 오류: {str(e)}"
            }
    
    def get_all_tools(self) -> List[McpTool]:
        """연결된 모든 서버의 도구 목록 반환"""
        all_tools = []
        for server_tools in self.tools.values():
            all_tools.extend(server_tools)
        return all_tools
    
    def get_server_info_for_prompt(self) -> str:
        """시스템 프롬프트에 포함할 MCP 서버 정보 생성"""
        if not self.connections:
            return "(No MCP servers currently connected)"
        
        server_descriptions = []
        for server_name, config in self.servers.items():
            if server_name in self.connections:
                tools_info = ""
                if server_name in self.tools and self.tools[server_name]:
                    tools_list = []
                    for tool in self.tools[server_name]:
                        schema_str = ""
                        if tool.input_schema:
                            schema_str = f"\n    Input Schema: {json.dumps(tool.input_schema, indent=2)}"
                        tools_list.append(f"- {tool.name}: {tool.description}{schema_str}")
                    tools_info = f"\n\n### Available Tools\n" + "\n\n".join(tools_list)
                
                cmd_info = f"{config.command}"
                if config.args:
                    cmd_info += f" {' '.join(config.args)}"
                
                server_descriptions.append(f"## {server_name} (`{cmd_info}`){tools_info}")
        
        return "\n\n".join(server_descriptions)
    
    async def cleanup(self):
        """모든 MCP 연결 정리"""
        for server_name, process in self.connections.items():
            try:
                process.terminate()
                await asyncio.sleep(1)
                if process.poll() is None:
                    process.kill()
                process.wait()
                print(f"🧹 MCP 서버 '{server_name}' 연결 정리됨")
            except Exception as e:
                print(f"⚠️  서버 '{server_name}' 정리 중 오류: {e}")
        
        self.connections.clear()
        self.servers.clear()
        self.tools.clear()
        self.resources.clear()

# ============================================================================
# 3. LLM API 클라이언트 - HTTP POST 방식
# ============================================================================

class LLMClient:
    """HTTP POST 방식으로 LLM API 호출하는 클라이언트"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate_response(self, messages: List[LLMMessage]) -> str:
        """LLM에게 메시지를 보내고 응답 받기"""
        if self.config.provider == "openai":
            return await self._call_openai_api(messages)
        elif self.config.provider == "anthropic":
            return await self._call_anthropic_api(messages)
        elif self.config.provider == "ollama":
            return await self._call_ollama_api(messages)
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {self.config.provider}")
    
    async def _call_openai_api(self, messages: List[LLMMessage]) -> str:
        """OpenAI API 호출"""
        url = self.config.base_url or "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    async def _call_anthropic_api(self, messages: List[LLMMessage]) -> str:
        """Anthropic Claude API 호출"""
        url = self.config.base_url or "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # 시스템 메시지와 일반 메시지 분리
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})
        
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": conversation_messages
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]
    
    async def _call_ollama_api(self, messages: List[LLMMessage]) -> str:
        """Ollama API 호출"""
        url = self.config.base_url or "http://localhost:11434/api/chat"
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]

# ============================================================================
# 4. 시스템 프롬프트 생성기
# ============================================================================

class SystemPromptGenerator:
    """MCP 정보를 포함한 시스템 프롬프트 생성"""
    
    def __init__(self, mcp_hub: McpHub):
        self.mcp_hub = mcp_hub
    
    def generate_system_prompt(self, mode: str = "assistant") -> str:
        """시스템 프롬프트 생성"""
        
        # 기본 역할 정의
        role_definition = self._get_role_definition(mode)
        
        # 도구 사용 지침
        tool_usage_section = self._get_tool_usage_section()
        
        # MCP 도구 설명
        mcp_tools_section = self._get_mcp_tools_section()
        
        # 연결된 MCP 서버 정보
        mcp_servers_section = self._get_mcp_servers_section()
        
        return f"""{role_definition}

{tool_usage_section}

{mcp_tools_section}

{mcp_servers_section}

## 규칙
1. 각 도구 사용 후 반드시 결과를 기다리세요.
2. MCP 도구 사용 시 정확한 서버 이름과 도구 이름을 사용하세요.
3. 도구 매개변수는 JSON 형식으로 정확히 입력하세요.
4. 오류 발생 시 사용자에게 명확하게 설명하세요.
"""
    
    def _get_role_definition(self, mode: str) -> str:
        """역할 정의"""
        if mode == "assistant":
            return """당신은 도움이 되는 AI 어시스턴트입니다. 다양한 도구를 사용하여 사용자의 요청을 수행할 수 있습니다."""
        elif mode == "code":
            return """당신은 전문적인 코딩 어시스턴트입니다. 코드 작성, 분석, 디버깅을 도와줍니다."""
        else:
            return """당신은 유능한 AI 어시스턴트입니다."""
    
    def _get_tool_usage_section(self) -> str:
        """도구 사용 지침"""
        return """## 도구 사용

다음 XML 형식으로 도구를 사용하세요:

<tool_name>
<parameter1>value1</parameter1>
<parameter2>value2</parameter2>
</tool_name>

예시:
<use_mcp_tool>
<server_name>weather-server</server_name>
<tool_name>get_weather</tool_name>
<arguments>{"city": "Seoul", "units": "celsius"}</arguments>
</use_mcp_tool>"""
    
    def _get_mcp_tools_section(self) -> str:
        """MCP 도구 설명"""
        return """## 사용 가능한 MCP 도구

### use_mcp_tool
MCP 서버의 도구를 실행합니다.
매개변수:
- server_name: (필수) MCP 서버 이름
- tool_name: (필수) 실행할 도구 이름  
- arguments: (필수) 도구에 전달할 매개변수 (JSON 형식)

### access_mcp_resource
MCP 서버의 리소스에 접근합니다.
매개변수:
- server_name: (필수) MCP 서버 이름
- uri: (필수) 접근할 리소스 URI"""
    
    def _get_mcp_servers_section(self) -> str:
        """연결된 MCP 서버 정보"""
        server_info = self.mcp_hub.get_server_info_for_prompt()
        
        return f"""## 연결된 MCP 서버

{server_info}

서버가 연결되어 있으면 `use_mcp_tool` 도구로 서버의 도구를 사용하고, `access_mcp_resource` 도구로 리소스에 접근할 수 있습니다."""

# ============================================================================
# 5. 도구 실행 처리기
# ============================================================================

class ToolExecutor:
    """LLM 응답에서 도구 호출을 파싱하고 실행"""
    
    def __init__(self, mcp_hub: McpHub):
        self.mcp_hub = mcp_hub
    
    async def parse_and_execute_tools(self, llm_response: str) -> List[Dict[str, Any]]:
        """LLM 응답에서 도구 호출을 파싱하고 실행"""
        import re
        
        results = []
        
        # use_mcp_tool 패턴 찾기
        mcp_tool_pattern = r'<use_mcp_tool>\s*<server_name>([^<]+)</server_name>\s*<tool_name>([^<]+)</tool_name>\s*<arguments>([^<]+)</arguments>\s*</use_mcp_tool>'
        
        for match in re.finditer(mcp_tool_pattern, llm_response, re.DOTALL):
            server_name = match.group(1).strip()
            tool_name = match.group(2).strip()
            arguments_str = match.group(3).strip()
            
            try:
                arguments = json.loads(arguments_str)
                result = await self.mcp_hub.call_tool(server_name, tool_name, arguments)
                
                results.append({
                    "type": "use_mcp_tool",
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "type": "use_mcp_tool",
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "error": str(e)
                })
        
        # access_mcp_resource 패턴 찾기
        mcp_resource_pattern = r'<access_mcp_resource>\s*<server_name>([^<]+)</server_name>\s*<uri>([^<]+)</uri>\s*</access_mcp_resource>'
        
        for match in re.finditer(mcp_resource_pattern, llm_response, re.DOTALL):
            server_name = match.group(1).strip()
            uri = match.group(2).strip()
            
            try:
                result = await self.mcp_hub.read_resource(server_name, uri)
                
                results.append({
                    "type": "access_mcp_resource",
                    "server_name": server_name,
                    "uri": uri,
                    "result": result
                })
                
            except Exception as e:
                results.append({
                    "type": "access_mcp_resource",
                    "server_name": server_name,
                    "uri": uri,
                    "error": str(e)
                })
        
        return results

# ============================================================================
# 6. 메인 대화 관리자
# ============================================================================

class ConversationManager:
    """전체 대화를 관리하는 메인 클래스"""
    
    def __init__(self, llm_config: LLMConfig):
        self.mcp_hub = McpHub()
        self.llm_client = LLMClient(llm_config)
        self.prompt_generator = SystemPromptGenerator(self.mcp_hub)
        self.tool_executor = ToolExecutor(self.mcp_hub)
        self.conversation_history: List[LLMMessage] = []
    
    async def setup_mcp_servers(self, server_configs: List[McpServerConfig]):
        """MCP 서버들 연결 설정"""
        print("🔧 MCP 서버 연결 중...")
        for config in server_configs:
            await self.mcp_hub.connect_server(config)
        
        print(f"✅ 총 {len(self.mcp_hub.connections)}개 MCP 서버 연결됨")
        
        # 연결된 도구들 출력
        all_tools = self.mcp_hub.get_all_tools()
        if all_tools:
            print(f"📚 사용 가능한 도구: {', '.join([tool.name for tool in all_tools])}")
    
    async def process_user_message(self, user_message: str) -> str:
        """사용자 메시지 처리"""
        print(f"👤 사용자: {user_message}")
        
        # 대화 히스토리에 사용자 메시지 추가
        self.conversation_history.append(LLMMessage(role="user", content=user_message))
        
        # 시스템 프롬프트 생성
        system_prompt = self.prompt_generator.generate_system_prompt()
        
        # LLM API 호출
        messages = [LLMMessage(role="system", content=system_prompt)] + self.conversation_history
        
        llm_response = await self.llm_client.generate_response(messages)
        print(f"🤖 LLM 응답: {llm_response}")
        
        # 도구 호출 파싱 및 실행
        tool_results = await self.tool_executor.parse_and_execute_tools(llm_response)
        
        final_response = llm_response
        
        # 도구 실행 결과가 있으면 처리
        if tool_results:
            print(f"🔧 {len(tool_results)}개 도구 실행됨")
            
            # 도구 실행 결과를 대화에 추가
            self.conversation_history.append(LLMMessage(role="assistant", content=llm_response))
            
            tool_results_text = ""
            for result in tool_results:
                if "error" in result:
                    tool_results_text += f"❌ 도구 실행 오류: {result['error']}\n"
                else:
                    tool_results_text += f"✅ 도구 실행 결과:\n{result['result']['content']}\n\n"
            
            # 도구 결과를 포함해서 다시 LLM 호출
            if tool_results_text.strip():
                self.conversation_history.append(LLMMessage(role="user", content=f"도구 실행 결과:\n{tool_results_text}"))
                
                messages = [LLMMessage(role="system", content=system_prompt)] + self.conversation_history
                final_response = await self.llm_client.generate_response(messages)
                print(f"🤖 최종 응답: {final_response}")
        
        # 대화 히스토리에 최종 응답 추가
        self.conversation_history.append(LLMMessage(role="assistant", content=final_response))
        
        return final_response

# ============================================================================
# 7. 사용 예제
# ============================================================================

async def main():
    """메인 실행 함수"""
    
    # LLM 설정
    llm_config = LLMConfig(
        provider="ollama",  # "openai", "anthropic", "ollama" 중 선택
        model="llama3.2",
        base_url="http://localhost:11434",  # Ollama 기본 URL
        temperature=0.7
    )
    
    # 대화 관리자 초기화
    manager = ConversationManager(llm_config)
            # command="a:\\simpledraw-mcp\\venv\\Scripts\\python.exe",
            # args=["a:\\simpledraw-mcp\\simpledraw.py"],  # 실제 MCP 서버 경로
    # MCP 서버 설정 (예: 날씨 서버)
    mcp_servers = [
        McpServerConfig(
            name="example-server",
            type=McpServerType.STDIO,
            command="a:\\mcpclient\\venv\\Scripts\\python.exe",
            args=["a:\\mcpclient\\example_mcp_server.py"],  # 간단한 MCP 서버
            timeout=10  # 10초 타임아웃
        ),
        # 추가 MCP 서버들...
    ]
    
    # MCP 서버 연결
    await manager.setup_mcp_servers(mcp_servers)
    
    # 대화 시작
    print("\n💬 MCP + LLM 대화 시작! (종료하려면 'quit' 입력)")
    print("=" * 50)
    
    while True:
        user_input = input("\n👤 당신: ")
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("👋 대화를 종료합니다.")
            break
        
        try:
            response = await manager.process_user_message(user_input)
            print(f"\n🤖 AI: {response}")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    # 정리
    await manager.mcp_hub.cleanup()
    await manager.llm_client.client.aclose()

# ============================================================================
# 8. Streamlit 챗봇 예제 (주석)
# ============================================================================

"""
Streamlit 챗봇으로 변환하는 예제:

```python
import streamlit as st
import asyncio

# 전역 변수로 ConversationManager 저장
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = None
    st.session_state.setup_complete = False

def load_mcp_config(config_file: str = "test.json") -> List[McpServerConfig]:
    \"\"\"JSON 설정 파일에서 MCP 서버 설정 로드\"\"\"
    import json
    import os
    
    if not os.path.exists(config_file):
        st.error(f"MCP 설정 파일을 찾을 수 없습니다: {config_file}")
        return []
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        mcp_servers = []
        for server_name, server_config in config_data.get("mcpServers", {}).items():
            mcp_servers.append(McpServerConfig(
                name=server_name,
                type=McpServerType.STDIO,
                command=server_config.get("command"),
                args=server_config.get("args", []),
                env=server_config.get("env"),
                timeout=server_config.get("timeout", 30)
            ))
        
        return mcp_servers
        
    except Exception as e:
        st.error(f"MCP 설정 파일 읽기 실패: {e}")
        return []

async def setup_mcp_system():
    \"\"\"MCP 시스템 초기화 (한 번만 실행)\"\"\"
    if st.session_state.setup_complete:
        return st.session_state.conversation_manager
    
    # LLM 설정
    llm_config = LLMConfig(
        provider="ollama",
        model="llama3.2", 
        base_url="http://localhost:11434",
        temperature=0.7
    )
    
    # 대화 관리자 초기화
    manager = ConversationManager(llm_config)
    
    # JSON 파일에서 MCP 서버 설정 로드
    mcp_servers = load_mcp_config("test.json")  # test.json 파일에서 설정 읽기
    
    if not mcp_servers:
        st.warning("연결할 MCP 서버가 없습니다. test.json 파일을 확인해주세요.")
        return None
    
    # MCP 서버 연결
    await manager.setup_mcp_servers(mcp_servers)
    
    st.session_state.conversation_manager = manager
    st.session_state.setup_complete = True
    return manager

async def chat_with_mcp(user_message: str) -> str:
    \"\"\"MCP 도구를 사용하여 사용자 메시지 처리\"\"\"
    manager = await setup_mcp_system()
    
    try:
        response = await manager.process_user_message(user_message)
        return response
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

def main_streamlit():
    \"\"\"Streamlit 메인 함수\"\"\"
    st.title("🤖 MCP + LLM 챗봇")
    st.caption("MCP 도구를 사용하는 AI 어시스턴트")
    
    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                # asyncio.run을 사용하여 비동기 함수 실행
                response = asyncio.run(chat_with_mcp(prompt))
                st.markdown(response)
        
        # AI 응답 저장
        st.session_state.messages.append({"role": "assistant", "content": response})

# Streamlit 실행
if __name__ == "__main__":
    # 일반 콘솔 모드
    asyncio.run(main())
    
    # Streamlit 모드로 실행하려면:
    # streamlit run python_mcp_llm_example.py
```

사용법:
1. 터미널에서: `streamlit run python_mcp_llm_example.py`
2. 브라우저에서 자동으로 열리는 페이지에서 채팅
3. MCP 도구들이 자동으로 사용됨 (시간 조회, 계산, 텍스트 분석, UUID 생성 등)

예제 입력:
- "현재 시간 알려줘"
- "2 + 3 * 4를 계산해줘" 
- "안녕하세요 텍스트를 분석해주세요"
- "UUID 하나 생성해줘"
"""

if __name__ == "__main__":
    asyncio.run(main()) 