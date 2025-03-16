import imgkit

mermaid_code = '''
flowchart TD
    %% User interaction
    User([User]) --> |"Submit Query"| QueryEndpoint["query.py\nsubmit_query()"]
    User --> |"Check Status"| StatusEndpoint["query.py\nget_query_status()"]
    User --> |"Real-time Updates"| WebsocketConn["websocket.py\nwebsocket_endpoint()"]
    User --> |"Retrieve Results"| ResultEndpoint["query.py\nget_query_result()"]
    
    %% API Layer
    subgraph API ["API Layer"]
        QueryEndpoint --> |"Validate"| QuerySchema["query.py\nQueryRequest"]
        StatusEndpoint --> QueryStatus["query.py\nQueryStatus"]
        ResultEndpoint --> QueryResponse["query.py\nQueryResponse"]
        WebsocketConn --> |"Manage Connection"| ConnManager["websocket.py\nConnectionManager.connect()"]
        
        QueryEndpoint --> |"Submit Task"| TaskSubmit["queue.py\nTaskQueue.submit_task()"]
        StatusEndpoint --> |"Check Status"| TaskStatus["queue.py\nTaskQueue.get_task_status()"]
        ResultEndpoint --> |"Get Result"| TaskResult["queue.py\nTaskQueue.get_task_result()"]
    end
    
    %% Asynchronous Processing
    subgraph Worker ["Worker Layer"]
        TaskSubmit --> WorkerExec["tasks.py\nexecute_workflow_task()"]
        WorkerExec --> WFExec["workflow.py\nWorkflowManager.execute_workflow()"]
        WFExec --> |"Progress Update"| UpdateProgress["tasks.py\nupdate_progress()"]
        UpdateProgress --> ConnManager
    end
    
    %% Workflow Execution 
    subgraph Workflow ["Workflow Layer"]
        WFExec --> ResearchWorkflow["workflow.py\nResearchWorkflow.execute()"]
        ResearchWorkflow --> WorkflowExecute["workflow.py\nWorkflow.execute()"]
        WorkflowExecute --> |"Step 1"| GetResearcher["manager.py\nAgentManager.get_agent()"]
        WorkflowExecute --> |"Step 2"| GetAnalyzer["manager.py\nAgentManager.get_agent()"]
        WorkflowExecute --> |"Step 3"| GetSummarizer["manager.py\nAgentManager.get_agent()"]
        
        GetResearcher --> |"Execute"| ExecuteResearcher["manager.py\nAgentManager.execute_agent()"]
        GetAnalyzer --> |"Execute"| ExecuteAnalyzer["manager.py\nAgentManager.execute_agent()"]
        GetSummarizer --> |"Execute"| ExecuteSummarizer["manager.py\nAgentManager.execute_agent()"]
    end
    
    %% Agent Layer
    subgraph Agents ["Agent Layer"]
        ExecuteResearcher --> ResearcherSafe["researcher.py\nResearcherAgent.safe_execute()"]
        ExecuteAnalyzer --> AnalyzerSafe["analyzer.py\nAnalyzerAgent.safe_execute()"]
        ExecuteSummarizer --> SummarizerSafe["summarizer.py\nSummarizerAgent.safe_execute()"]
        
        ResearcherSafe --> |"Validate"| ResearcherValidate["base.py\nBaseAgent.validate_input()"]
        ResearcherSafe --> ResearcherExec["researcher.py\nResearcherAgent.execute()"]
        
        AnalyzerSafe --> |"Validate"| AnalyzerValidate["base.py\nBaseAgent.validate_input()"]
        AnalyzerSafe --> AnalyzerExec["analyzer.py\nAnalyzerAgent.execute()"]
        
        SummarizerSafe --> |"Validate"| SummarizerValidate["base.py\nBaseAgent.validate_input()"]
        SummarizerSafe --> SummarizerExec["summarizer.py\nSummarizerAgent.execute()"]
    end
    
    %% Tool Layer
    subgraph Tools ["Tool Layer"]
        ResearcherExec --> |"Web Search"| SerperSearch["serper.py\nSerperTool.search()"]
        ResearcherExec --> |"Process Results"| SerperProcess["serper.py\nSerperTool._process_results()"]
        
        AnalyzerExec --> |"Embedding"| JinaEmbed["jina.py\nJinaTool.generate_embeddings()"]
        AnalyzerExec --> |"Document Processing"| JinaProcess["jina.py\nJinaTool.process_documents()"]
        
        SummarizerExec --> |"Summarize"| OpenAISummarize["openai.py\nOpenAITool.summarize()"]
        OpenAISummarize --> OpenAIProcess["openai.py\nOpenAITool.process()"]
    end
    
    %% Database Operations
    subgraph Database ["Database Layer"]
        WorkflowExecute --> |"Save Results"| SaveResult["results.py\nCRUDResult.save_result()"]
        SaveResult --> DBSession["session.py\nSessionLocal()"]
        DBSession --> DBModels["models.py\nResult"]
        
        ResearcherExec --> |"Record Execution"| SaveExecution["results.py\nCRUDResult.save_agent_execution()"]
        SaveExecution --> AgentExecModel["models.py\nAgentExecution"]
        
        TaskResult --> |"Fetch Results"| GetResult["results.py\nCRUDResult.get_by_task_id()"]
    end
    
    %% Monitoring Layer
    subgraph Monitoring ["Monitoring Layer"]
        QueryEndpoint --> |"Start Trace"| CreateTrace["tracer.py\nLangfuseTracer.create_trace()"]
        ResearcherExec --> |"Create Span"| CreateSpan["tracer.py\nLangfuseTracer.span()"]
        ResearcherExec --> |"Log Execution"| LogExecution["base.py\nBaseAgent.log_execution()"]
        
        SerperSearch --> |"Log Event"| LogEvent["tracer.py\nLangfuseTracer.log_event()"]
        OpenAIProcess --> |"Log Generation"| LogGeneration["tracer.py\nLangfuseTracer.log_generation()"]
        
        QueryEndpoint --> |"Record Metrics"| PrometheusMiddleware["metrics.py\nPrometheusMiddleware.dispatch()"]
        UpdateProgress --> |"Track Progress"| TrackWorkflow["metrics.py\ntrack_workflow_execution()"]
        LogExecution --> |"Track Execution"| TrackAgent["metrics.py\ntrack_agent_execution()"]
        
        LogExecution --> |"Structured Logging"| JSONFormatter["logging.py\nJSONFormatter.format()"]
    end
    
    %% Result Flow Back to User
    SaveResult --> ResultDB[(Database)]
    
    WorkerExec --> |"Task Complete"| TaskComplete["celery_app.py\ntask_postrun_handler()"]
    TaskComplete --> |"Send Final Update"| FinalUpdate["websocket.py\nConnectionManager.broadcast()"]
    FinalUpdate --> WebsocketConn
    
    ResultEndpoint --> |"Return Results"| UserResponse["response.py\nStandardResponse"]
    UserResponse --> User
    
    %% Legend & Styling
    classDef userflow fill:#f96,stroke:#333,stroke-width:1px
    classDef api fill:#bbf,stroke:#333,stroke-width:1px
    classDef worker fill:#bfb,stroke:#333,stroke-width:1px
    classDef workflow fill:#9cf,stroke:#333,stroke-width:1px
    classDef agent fill:#fcb,stroke:#333,stroke-width:1px
    classDef tool fill:#f9f,stroke:#333,stroke-width:1px
    classDef database fill:#fd9,stroke:#333,stroke-width:1px
    classDef monitoring fill:#ddd,stroke:#333,stroke-width:1px
    
    class User,QueryEndpoint,StatusEndpoint,WebsocketConn,ResultEndpoint userflow
    class QuerySchema,QueryStatus,QueryResponse,ConnManager,TaskSubmit,TaskStatus,TaskResult api
    class WorkerExec,UpdateProgress,WFExec worker
    class ResearchWorkflow,WorkflowExecute,GetResearcher,GetAnalyzer,GetSummarizer,ExecuteResearcher,ExecuteAnalyzer,ExecuteSummarizer workflow
    class ResearcherSafe,AnalyzerSafe,SummarizerSafe,ResearcherExec,AnalyzerExec,SummarizerExec,ResearcherValidate,AnalyzerValidate,SummarizerValidate agent
    class SerperSearch,SerperProcess,JinaEmbed,JinaProcess,OpenAISummarize,OpenAIProcess tool
    class SaveResult,GetResult,DBSession,DBModels,SaveExecution,AgentExecModel,ResultDB database
    class CreateTrace,CreateSpan,LogExecution,LogEvent,LogGeneration,PrometheusMiddleware,TrackWorkflow,TrackAgent,JSONFormatter monitoring
'''

# Convert the Mermaid diagram to an image using imgkit
options = {"format": "png"}
imgkit.from_string(mermaid_code, "flowchart.png", options=options)
