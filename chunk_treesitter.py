'''
splits python files by function/class boundaries using tree-sitter
instead of naive chunking by character count.

'''

from dataclasses import dataclass
from urllib import parse
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())  # creates a Language object 
                                             # from the Python grammar
parser = Parser(PY_LANGUAGE)                 # creates a parser that can
                                             # parse Python code


@dataclass
class CodeChunk:
    text: str
    source_file: str
    chunk_type: str  # func, class, module-level
    name: str        # function or class name or empty
    start_line: int
    end_line: int


def extract_chunks_from_file(filepath: str) -> list[CodeChunk]:

    '''
    parse one python file and pull out top-level 
    functions and classes as seperate chunks.

    '''

    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()

    if not source_code.strip():  # skip empty files
        return []
    
    tree = parser.parse(bytes(source_code, "utf8"))
    root = tree.root_node
    source_lines = source_code.split('\n')

    chunks = []
    covered_lines = set()  # to track lines already included in chunks

    for node in root.children: 
        if node.type in ('function_definition', 'class_definition' ):

            start_line, end_line = node.start_point[0], node.end_point[0]
            chunk_text = '\n'.join(source_lines[start_line:end_line+1])

            # extract function/class name for better metadata
            name = "unknown"
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode('utf-8')
                    break

            
            chunks.append(CodeChunk(
                text=chunk_text,
                source_file=filepath,
                chunk_type='function' if node.type == 'function_definition' else 'class',
                name=name,
                start_line=start_line+1,
                end_line=end_line+1
            ))

            covered_lines.update(range(start_line, end_line+1))


    leftover_lines = [
        (i,line) for i,line in enumerate(source_lines)
        if i not in covered_lines and line.strip()
    ]

    if leftover_lines:

        leftover_text = '\n'.join(line for _, line in leftover_lines)
        if leftover_text.strip():   # only add if its non empty
            chunks.append(CodeChunk(
                text=leftover_text,
                source_file=filepath,
                chunk_type="module-level",
                name="module",
                start_line=leftover_lines[0][0]+1,
                end_line=leftover_lines[-1][0]+1
            ))

    
    return chunks


def chunk_repo_treesitter(file_paths: list[str]) -> list[CodeChunk]:

    '''
    takes a list of file paths and returns a
    list of CodeChunk objects for each file

    '''

    all_chunks = []

    for path in file_paths:
        
        try:
            file_chunks = extract_chunks_from_file(path)
            all_chunks.extend(file_chunks)

        except Exception as e:
            print(f"Error processing {path}: {e}. Skipping this file.")

    return all_chunks


if __name__ == "__main__":

    from clone_repo import find_python_files, CLONE_DIR
    
    files = find_python_files(CLONE_DIR)
    chunks = chunk_repo_treesitter(files)

    print(f"Total Chunks: {len(chunks)}")

    func_chunks = [c for c in chunks if c.chunk_type == 'function']
    class_chunks = [c for c in chunks if c.chunk_type == 'class']
    module_chunks = [c for c in chunks if c.chunk_type == 'module-level']

    print(f"Function Chunks: {len(func_chunks)}")
    print(f"Class Chunks: {len(class_chunks)}")
    print(f"Module-level Chunks: {len(module_chunks)}")

    # find a chunk with Depends in it (for example)
    # and print it out so you can see that it's not cut off 
    # for c in chunks:
    #     if c.name == "verify_jwt_token" or (c.chunk_type == "function" and "depend" in c.name.lower()):
    #         print(f"\n--- Example: {c.name} ({c.chunk_type}) from {c.source_file} lines {c.start_line}-{c.end_line} ---")
    #         print(c.text[:500])
    #         break
    # else:
    #     # fallback: just show any complete function so you can SEE it's not cut off 
    #     print(f"\n--- Example function chunk: {func_chunks[2].name} ---")
    #     print(f"Source: {func_chunks[2].source_file} lines {func_chunks[2].start_line}-{func_chunks[2].end_line}")
    #     print(func_chunks[2].text[:500])