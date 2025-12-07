"""
Simple mock LLM server for testing eval.py without external dependencies.
Responds to /v1/chat/completions with random answers from the dataset.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import time
import random


class MockLLMHandler(BaseHTTPRequestHandler):
    """Mock OpenAI-compatible chat completions endpoint"""
    
    # Sample answers for common questions (simple mock data)
    MOCK_ANSWERS = {
        "basketball": "Basketball",
        "sport": "Basketball",
        "miami heat": "Basketball",
        "stage musical": "The Phantom of the Opera",
        "sequel": "The Phantom of the Opera",
        "loves dies": "The Phantom of the Opera",
        "denali": "Alaska",
        "national park": "Alaska",
        "us state": "Alaska",
        "film": "1994",
        "year": "1994",
        "released": "1994",
        "forrest gump": "1994",
        "capital": "Washington, D.C.",
        "united states": "Washington, D.C.",
        "president": "George Washington",
        "washington": "Washington, D.C.",
    }
    
    def do_POST(self):
        """Handle POST requests to /v1/chat/completions"""
        if self.path != "/v1/chat/completions":
            self.send_error(404, "Not found")
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode('utf-8'))
            
            # Extract the user's question from messages
            messages = request_data.get('messages', [])
            question = ""
            for msg in messages:
                if msg.get('role') == 'user':
                    question = msg.get('content', '')
                    break
            
            # Generate a mock answer based on keywords in the question
            answer = self._generate_answer(question)
            
            # Return OpenAI-compatible response
            response = {
                "id": "mock-" + str(int(time.time())),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": answer
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20
                }
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"Error: {e}")
            self.send_error(500, str(e))
    
    def _generate_answer(self, question: str) -> str:
        """Generate a mock answer based on question keywords"""
        question_lower = question.lower()
        
        # Check for keyword matches
        for keyword, answer in self.MOCK_ANSWERS.items():
            if keyword in question_lower:
                return answer
        
        # Default answer if no keywords match
        default_answers = ["Unknown", "I don't know", "Not sure", "Cannot determine"]
        return random.choice(default_answers)
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def start_server(host='localhost', port=8000):
    """Start the mock LLM server"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, MockLLMHandler)
    print(f"Mock LLM server started at http://{host}:{port}/v1/chat/completions")
    print("Press Ctrl+C to stop the server\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    start_server()
