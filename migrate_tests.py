import os
import sys
import libcst as cst
from libcst.metadata import PositionProvider

class AsyncTestTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self):
        super().__init__()
        self.in_test_func = False
        self.made_changes = False
        self.func_stack = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        is_test = node.name.value.startswith("test_") and not node.asynchronous
        self.func_stack.append({'is_test': is_test, 'made_changes': False})
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        state = self.func_stack.pop()
        if state['is_test'] and state['made_changes']:
            return updated_node.with_changes(asynchronous=cst.Asynchronous())
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        if not self.func_stack or not self.func_stack[-1]['is_test']:
            return updated_node

        func = updated_node.func
        if isinstance(func, cst.Attribute):
            name = func.attr.value
            replacements = {
                "invoke": "ainvoke",
                "generate": "agenerate",
                "batch": "abatch",
                "similarity_search": "asimilarity_search",
                "add_texts": "aadd_texts",
                "add_documents": "aadd_documents",
                "get_documents": "aget_documents",
                "batch_search": "abatch_search",
                "text_search": "atext_search",
                "delete": "adelete",
            }
            if name in replacements:
                self.func_stack[-1]['made_changes'] = True
                new_attr = func.with_changes(attr=cst.Name(replacements[name]))
                new_call = updated_node.with_changes(func=new_attr)
                return cst.Await(expression=new_call)
        
        # Handle time.sleep -> asyncio.sleep
        if isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name):
            if func.value.value == "time" and func.attr.value == "sleep":
                self.func_stack[-1]['made_changes'] = True
                new_attr = func.with_changes(value=cst.Name("asyncio"), attr=cst.Name("sleep"))
                new_call = updated_node.with_changes(func=new_attr)
                return cst.Await(expression=new_call)

        return updated_node


def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = cst.parse_module(source)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return

    transformer = AsyncTestTransformer()
    modified_tree = tree.visit(transformer)

    if modified_tree.code != source:
        # Add import asyncio if it's not there and we added asyncio.sleep
        code = modified_tree.code
        if "asyncio.sleep" in code and "import asyncio" not in code:
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if not line.startswith("from __future__") and not line.startswith("\"\"\""):
                    lines.insert(i, "import asyncio")
                    break
            code = "\n".join(lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Modified {filepath}")

def main():
    dirs_to_check = [
        "libs/community/tests/integration_tests",
        "libs/genai/tests/integration_tests",
        "libs/vertexai/tests/integration_tests",
    ]
    for d in dirs_to_check:
        for root, _, files in os.walk(d):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    process_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
