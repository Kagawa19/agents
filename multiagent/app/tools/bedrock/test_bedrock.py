import asyncio
import os
from dotenv import load_dotenv
from multiagent.app.tools.bedrock.bedrock_tool import BedrockTool
from multiagent.app.tools.bedrock.claude_reason import ClaudeReason
from multiagent.app.tools.bedrock.claude_summarize import ClaudeSummarize

# Load environment variables from .env file
load_dotenv()

async def test_bedrock():
    print("Initializing Bedrock client...")
    
    # Get credentials from environment variables
    config = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1")
    }
    
    # Initialize Bedrock tool
    bedrock_tool = BedrockTool(config)
    
    if not bedrock_tool.is_available():
        print("Error: Bedrock client initialization failed.")
        return
    
    print("Bedrock client initialized successfully!")
    
    # Test Claude reasoning
    print("\nTesting reasoning capability...")
    claude_reason = ClaudeReason()
    reasoning_result = await claude_reason.reason(
        client=bedrock_tool.client,
        query="Explain quantum computing in simple terms.",
        reasoning_type="step_by_step"
    )
    
    print("\nReasoning Result:")
    if "error" in reasoning_result:
        print(f"Error: {reasoning_result['error']}")
    else:
        print(f"Step count: {reasoning_result.get('step_count', 0)}")
        print("First step:", reasoning_result.get("steps", ["No steps"])[0])
        
    # Test Claude summarization
    print("\nTesting summarization capability...")
    claude_summarize = ClaudeSummarize()
    text_to_summarize = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and display 
    "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by 
    major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    """
    
    summary_result = await claude_summarize.summarize(
        client=bedrock_tool.client,
        text=text_to_summarize,
        length="short",
        style="concise"
    )
    
    print("\nSummarization Result:")
    if "error" in summary_result:
        print(f"Error: {summary_result['error']}")
    else:
        print(summary_result.get("summary", "No summary generated"))

if __name__ == "__main__":
    asyncio.run(test_bedrock())