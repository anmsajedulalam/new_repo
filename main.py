from agent import build_graph
import os
import sys

def main():
    print("Initializing Agentic RAG System...")
    try:
        app = build_graph()
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    print("\n" + "="*50)
    print("Welcome to the Agentic RAG QA System")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("="*50 + "\n")

    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Session ID: {thread_id}")

    while True:
        try:
            user_input = input("Question: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            # Feature: Dynamic Ingestion
            if user_input.startswith("/add"):
                file_path = user_input.split(" ", 1)[1].strip()
                print(f"Adding file: {file_path}...")
                from ingest import add_single_file
                result_msg = add_single_file(file_path)
                print(result_msg)
                print("-" * 30)
                continue
            
            if not user_input.strip():
                continue

            print("\nThinking...")
            # We explicitly pass 'messages' so the memory saver can append to it. 
            # HumanMessage is needed here so LangGraph knows who said what in the history.
            from langchain_core.messages import HumanMessage
            
            inputs = {
                "question": user_input,
                "messages": [HumanMessage(content=user_input)] 
            }
            
            # Pass config for thread persistence
            result = app.invoke(inputs, config=config)
            
            # Extract final response
            messages = result["messages"]
            final_response = messages[-1].content
            
            print(f"\nAnswer: {final_response}\n")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    main()
