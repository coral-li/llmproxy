#!/usr/bin/env python
"""Interactive demo for LLMProxy testing"""

import argparse
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from test_proxy_config import get_test_config
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
from llmproxy.config.utils import get_proxy_url, get_proxy_base_url


class ProxyDemo:
    def __init__(self, proxy_url: str = None):
        """Initialize demo with proxy URL"""
        if proxy_url is None:
            proxy_url = get_proxy_url()
        self.proxy_url = proxy_url
        self.base_url = f"{proxy_url}/v1"
        self.console = Console()
        self.test_config = get_test_config()
        
    def show_welcome(self):
        """Display welcome message"""
        welcome_text = f"""
[bold cyan]LLMProxy Interactive Demo[/bold cyan]

This demo will show you how LLMProxy works by:
1. Making requests to your configured models
2. Demonstrating load balancing
3. Showing caching behavior
4. Displaying endpoint statistics

[yellow]Proxy URL:[/yellow] {self.proxy_url}
[yellow]Test Model:[/yellow] {self.test_config['model']}
        """
        self.console.print(Panel(welcome_text, title="Welcome", border_style="cyan"))
        
    def demo_basic_request(self):
        """Demonstrate basic request"""
        self.console.print("\n[bold]1. Basic Request Demo[/bold]")
        self.console.print("Making a simple chat completion request...\n")
        
        # Show the code
        code = f'''client = OpenAI(
    base_url="{self.base_url}",
    api_key="dummy"  # LLMProxy handles authentication
)

response = client.chat.completions.create(
    model="{self.test_config['model']}",
    messages=[
        {{"role": "system", "content": "You are a helpful assistant."}},
        {{"role": "user", "content": "Say 'Hello from LLMProxy!' in exactly 5 words."}}
    ],
    temperature=0,
    max_tokens=20
)'''
        
        self.console.print(Syntax(code, "python", theme="monokai"))
        
        # Execute it
        try:
            client = OpenAI(base_url=self.base_url, api_key="dummy")
            response = client.chat.completions.create(
                model=self.test_config['model'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello from LLMProxy!' in exactly 5 words."}
                ],
                temperature=0,
                max_tokens=20
            )
            
            self.console.print(f"\n[green]Response:[/green] {response.choices[0].message.content}")
            self.console.print(f"[dim]Model used: {response.model}[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Error:[/red] {e}")
            
    def demo_caching(self):
        """Demonstrate caching behavior"""
        self.console.print("\n[bold]2. Caching Demo[/bold]")
        self.console.print("Making the same request twice to show caching...\n")
        
        client = OpenAI(base_url=self.base_url, api_key="dummy")
        
        # Fixed message for caching
        messages = [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ]
        
        import time
        
        # First request
        self.console.print("[yellow]First request (will hit backend):[/yellow]")
        start = time.time()
        try:
            response1 = client.chat.completions.create(
                model=self.test_config['model'],
                messages=messages,
                temperature=0,  # Important for caching
                max_tokens=10
            )
            time1 = (time.time() - start) * 1000
            self.console.print(f"Response: {response1.choices[0].message.content}")
            self.console.print(f"Time: {time1:.0f}ms\n")
        except Exception as e:
            self.console.print(f"[red]Error:[/red] {e}")
            return
            
        # Second request (should be cached)
        self.console.print("[yellow]Second request (should be cached):[/yellow]")
        start = time.time()
        try:
            response2 = client.chat.completions.create(
                model=self.test_config['model'],
                messages=messages,
                temperature=0,
                max_tokens=10
            )
            time2 = (time.time() - start) * 1000
            self.console.print(f"Response: {response2.choices[0].message.content}")
            self.console.print(f"Time: {time2:.0f}ms")
            
            if time2 < time1 / 2:
                self.console.print("[green]✓ Cache hit! Second request was much faster.[/green]")
            else:
                self.console.print("[yellow]⚠ Cache might not be working as expected.[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]Error:[/red] {e}")
            
    def demo_load_balancing(self):
        """Show load balancing across endpoints"""
        self.console.print("\n[bold]3. Load Balancing Demo[/bold]")
        self.console.print("Making multiple requests to see load distribution...\n")
        
        # Create a table showing the request flow
        table = Table(title="Request Flow Visualization")
        table.add_column("Step", style="cyan")
        table.add_column("Description", style="white")
        
        table.add_row("1. Client Request", f"Client sends request to {self.proxy_url}")
        table.add_row("2. Load Balancer", "Proxy selects endpoint based on weights")
        table.add_row("3. Backend Call", "Request forwarded to selected endpoint")
        table.add_row("4. Response", "Response returned to client")
        
        self.console.print(table)
        
        # Check stats
        import httpx
        try:
            response = httpx.get(f"{self.proxy_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                
                self.console.print("\n[bold]Current Endpoint Statistics:[/bold]")
                for model_group, endpoints in stats.get('endpoints', {}).items():
                    if endpoints:
                        table = Table(title=f"Model Group: {model_group}")
                        table.add_column("Endpoint", style="cyan")
                        table.add_column("Status", style="green")
                        table.add_column("Requests", style="yellow")
                        table.add_column("Success Rate", style="magenta")
                        
                        for ep in endpoints:
                            table.add_row(
                                ep.get('base_url', 'Unknown')[:50] + "...",
                                ep.get('status', 'Unknown'),
                                str(ep.get('total_requests', 0)),
                                f"{ep.get('success_rate', 0):.1f}%"
                            )
                        
                        self.console.print(table)
                        
        except Exception as e:
            self.console.print(f"[yellow]Could not fetch stats: {e}[/yellow]")
            
    def run_interactive_mode(self):
        """Run in interactive mode"""
        self.console.print("\n[bold cyan]Interactive Mode[/bold cyan]")
        self.console.print("You can now make custom requests to the proxy.\n")
        
        client = OpenAI(base_url=self.base_url, api_key="dummy")
        
        while True:
            try:
                user_input = self.console.input("\n[bold]Enter your message (or 'quit' to exit):[/bold] ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                self.console.print("[dim]Sending request...[/dim]")
                
                response = client.chat.completions.create(
                    model=self.test_config['model'],
                    messages=[
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                self.console.print(f"\n[green]Response:[/green] {response.choices[0].message.content}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error:[/red] {e}")
                
        self.console.print("\n[yellow]Goodbye![/yellow]")
        
    def run(self, interactive=False):
        """Run the demo"""
        self.show_welcome()
        
        if not interactive:
            # Run all demos
            self.demo_basic_request()
            input("\nPress Enter to continue...")
            
            self.demo_caching()
            input("\nPress Enter to continue...")
            
            self.demo_load_balancing()
            
            self.console.print("\n[bold green]Demo complete![/bold green]")
            self.console.print("Run with --interactive flag for interactive mode.")
        else:
            self.run_interactive_mode()


def main():
    parser = argparse.ArgumentParser(description="LLMProxy Interactive Demo")
    parser.add_argument("--proxy-url", default=None,
                       help="Proxy URL (default: read from config)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # If proxy URL not provided, it will be read from config
    demo = ProxyDemo(proxy_url=args.proxy_url)
    demo.run(interactive=args.interactive)


if __name__ == "__main__":
    main() 