from fastmcp import FastMCP
import html2markdown

class HTMLToMarkdownTool(FastMCP):
    def __init__(self):
        super().__init__("html-to-markdown")

    def execute(self, html_content):
        return html2markdown.convert(html_content)

# Register the tool
HTMLToMarkdownTool()