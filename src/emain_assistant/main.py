from fastapi import FastAPI, HTTPException
import uvicorn
import uuid 
from typing import Dict, List, Any
from schemas import ProcessEmailRequest,ProcessEmailResponse, ProcessEmailHITLRequest, ProcessEmailHITLResponse,InterruptInfo
from agent import process_email
from agent_hitl import compiled_email_assistant_hitl
from langgraph.types import Command


app = FastAPI(
    title="Email Assistant API",
    description="A complex email assistant built with LangGraph and FastAPI",
    version="1.0.0"
)



@app.get("/")
async def root() -> Dict[str, str]:
    """Basic Root check endpoint."""
    return {"message": "Email Assistant API is running!"}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "email-assistant"}



def _get_allowed_actions(config: Dict[str, bool]) -> list[str]:
    actions = []
    if config.get("allow_accept", False):
        actions.append("accept")
    if config.get("allow_edit", False):
        actions.append("edit")
    if config.get("allow_ignore", False):
        actions.append("ignore")
    if config.get("allow_respond", False):
        actions.append("respond")
    return actions


def _extract_final_result(state: Dict[str, Any]) -> ProcessEmailResponse:
    classification = state.get("classification_decision", "respond")

    response_text = "No response generated"
    reasoning = f"Email classified as: {classification}"

    messages = state.get("messages", [])

    for message in reversed(messages):
        if getattr(message, 'tool_call_id', None) is not None:
            content = str(message.content)
            if "Email sent" in content or "Meeting scheduled" in content:
                response_text = content
                break
    
    return ProcessEmailResponse(
        classification=classification,
        response=response_text,
        reasoning=reasoning
    )




@app.post("/process-email", response_model=ProcessEmailResponse)
def process_email_endpoint(request: ProcessEmailRequest)-> ProcessEmailResponse:

    try:
        email_dict = {
            "author": request.email.author,
            "to": request.email.to,
            "subject": request.email.subject,
            "email_thread": request.email.email_thread
        }

        result = process_email(email_dict)

        return ProcessEmailResponse(
            classification=result["classification"],
            response=result["response"],
            reasoning=result["reasoning"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing email: {str(e)}"
        )



@app.post("/process-email-hitl", response_model=ProcessEmailHITLResponse)
async def process_email_hitl_endpoint(request: ProcessEmailHITLRequest) -> ProcessEmailHITLResponse:
    try:        
        is_resume = request.thread_id is not None and request.human_response is not None
        is_new = request.email is not None and request.thread_id is None
        
        if not is_resume and not is_new:
            raise HTTPException(
                status_code=400,
                detail="Either provide `email` for new workflow or `thread_id` + `human_response` for resume"
            )

        thread_id = request.thread_id if is_resume else str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        if is_new:
            email_dict = {
                "author": request.email.author,
                "to": request.email.to,
                "subject": request.email.subject,
                "email_thread": request.email.email_thread
            }
            
            for chunk in compiled_email_assistant_hitl.stream(
                {"email_input": email_dict}, 
                config=config
            ):
                if "__interrupt__" in chunk:
                    interrupt_data = chunk["__interrupt__"][0].value[0]
                    
                    return ProcessEmailHITLResponse(
                        status="interrupted",
                        thread_id=thread_id,
                        interrupt=InterruptInfo(
                            action=interrupt_data["action_request"]["action"],
                            args=interrupt_data["action_request"]["args"],
                            description=interrupt_data["description"],
                            allowed_actions=_get_allowed_actions(interrupt_data["config"])
                        )
                    )
                            
            complete_state = compiled_email_assistant_hitl.get_state(config)
            if complete_state and complete_state.values:
                result = _extract_final_result(complete_state.values)
                return ProcessEmailHITLResponse(
                    status="completed",
                    thread_id=thread_id,
                    result=result
                )
        
        else:

            try:
                state = compiled_email_assistant_hitl.get_state(config)
                if not state or not state.values:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Thread {thread_id} not found or has no saved state"
                    )
                
                if not state.next:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Thread {thread_id} workflow already completed. Cannot resume."
                    )
                
                human_response = request.human_response
                resume_command = Command(resume=[{
                    "type": human_response.type,
                    "args": human_response.args or {}
                }])
                
                for chunk in compiled_email_assistant_hitl.stream(
                    resume_command,
                    config=config
                ):
                    if "__interrupt__" in chunk:
                        interrupt_data = chunk["__interrupt__"][0].value[0]
                        
                        return ProcessEmailHITLResponse(
                            status="interrupted",
                            thread_id=thread_id,
                            interrupt=InterruptInfo(
                                action=interrupt_data["action_request"]["action"],
                                args=interrupt_data["action_request"]["args"],
                                description=interrupt_data["description"],
                                allowed_actions=_get_allowed_actions(interrupt_data["config"])
                            )
                        )
                                    

                complete_state = compiled_email_assistant_hitl.get_state(config)
                if complete_state and complete_state.values:
                    result = _extract_final_result(complete_state.values)
                    return ProcessEmailHITLResponse(
                        status="completed",
                        thread_id=thread_id,
                        result=result
                    )
                    
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                raise HTTPException(status_code=400, detail=f"Failed to resume thread: {str(e)}")
        
        raise HTTPException(status_code=500, detail="Unexpected workflow state")
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing HITL email: {str(e)}"
        )

@app.get("/process-email-hitl/{thread_id}")
async def get_hitl_thread_state(thread_id: str) -> Dict[str, Any]:
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = compiled_email_assistant_hitl.get_state(config)
        
        if not state or not state.values:
            raise HTTPException(
                status_code=404,
                detail=f"Thread {thread_id} not found"
            )
        
        classification = None
        if "classification_decision" in state.values:
            classification = state.values["classification_decision"]
        elif "triage_interrupt_handler" in state.values and isinstance(state.values["triage_interrupt_handler"], dict):
            if "classification_decision" in state.values["triage_interrupt_handler"]:
                classification = state.values["triage_interrupt_handler"]["classification_decision"]
        
        return {
            "thread_id": thread_id,
            "state": state.values,
            "classification": classification,
            "status": "interrupted" if state.next else "completed",
            "next_nodes": list(state.next) if state.next else []
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving thread state: {str(e)}"
        )









if __name__=="__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )