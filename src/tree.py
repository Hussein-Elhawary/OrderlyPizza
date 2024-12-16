import re
from abc import ABC, abstractmethod
from anytree import AnyNode, RenderTree
from preprocess import *

# class TopSemanticTree(SemanticTree):
#     def __init__(self, *, flat_string=None, tree_rep=None, root_symbol=None, children=None):
#         super(TopSemanticTree, self).__init__(flat_string=flat_string, tree_rep=tree_rep, root_symbol=root_symbol,
#                                               children=children)

#     def children(self):
#         '''
#         :return: (List) Return a list of TopSemanticTree objects that are children of `self` 
#         '''
#         return [TopSemanticTree(tree_rep=c) for c in self.tree_rep.children]

#     @classmethod
#     def get_semantics_only_tree(cls, tree_rep):
#         '''
#         Returns a class object by removing the non-semantic nodes from its tree 
#         representation. 
        
#         :param tree_rep: (AnyNode) A tree representation

#         :return: (TopSemanticTree) A tree class object with the non-semantic nodes removed.
#         '''
#         tree_rep_ = cls.remove_non_semantic_nodes(tree_rep)
#         return cls(tree_rep=tree_rep_)

#     @staticmethod
#     def remove_non_semantic_nodes(tree_rep):
#         '''
#         Method functionally removes the non-semantic nodes from a tree representation.

#         :param: (AnyNode) Pointer to the input tree.
#         :return: (AnyNode) Pointer to a new tree carrying only semantic nodes.

def linearized_rep_to_tree_rep(flat_string):
    '''
    Get the tree representation for flat input string `flat_string`
    Example input string:
    "(ORDER can i have (PIZZAORDER (NUMBER a ) (SIZE large ) (TOPPING bbq pulled pork ) ) please )"
    Invalid flat strings include those with misplaced brackets, mismatched brackets,
    or semantic nodes with no children
    :param flat_string: (str) input flat string to construct a tree, if possible.
    :raises ValueError: when s is not a valid flat string
    :raises IndexError: when s is not a valid flat string
    
    :return: (AnyNode) returns a pointer to a tree node.
    '''
    # Keep track of all the semantics in the input string.
    semantic_stack = [AnyNode(id='Start')]
    for token in flat_string.split():
        if '(' in token:
            # if token.strip('(') == 'ORDER':
            #     continue
            node = AnyNode(id=token.strip('('), parent=semantic_stack[-1])
            semantic_stack.append(node)
        elif token == ')':
            # If the string is not valid an error will be thrown here.
            # E.g. (PIZZAORDER (SIZE LARGE ) ) ) ) ) ) )
            try:
                # If there are no children within this semantic node, throw an error
                # E.g. (PIZZAORDER (SIZE LARGE ) (NOT ) )
                if not semantic_stack[-1].children:
                    raise Exception("Semantic node with no children")
                semantic_stack.pop()
            except Exception as e:
                raise IndexError(e) from e
        else:
            AnyNode(id=token, parent=semantic_stack[-1])
    # If there are more than one elements in semantic stack, that means
    # the input string is malformed, i.e. it cant be used to construct a tree
    if len(semantic_stack) > 1:
        raise ValueError()
    semantic_stack[-1] = semantic_stack[0].children[0]
    semantic_stack[-1].parent = None
    return semantic_stack[-1]

def traverse_tree(node: AnyNode, tags=None):
    '''
    Given a tree, traverse it and return a list of nodes in the order of traversal.
    :param node: (AnyNode) A node to start traversal from
    :param tags: (dict) A dictionary to store node ids and their parent ids
    :return: (dict) A dictionary of nodes and their parent ids
    '''
    if tags is None:
        tags = dict()
    
    tags[node.id] = node.parent.id if node.parent else None
    
    for child in node.children:
        traverse_tree(child, tags)
    
    return tags


def parse_tc(train_SRC,train_TOP):
    train_SRC = "i'd like a pizza with banana pepper grilled chicken and white onions without thin crust"
    train_TOP = "(ORDER i'd like (PIZZAORDER (NUMBER a ) pizza with (TOPPING banana pepper ) (TOPPING grilled chicken ) and (TOPPING white onions ) without (NOT (STYLE thin crust ) ) ) )"

    def parse_sexp(s):
        s = s.replace('(', ' ( ').replace(')', ' ) ')
        tokens = s.split()
        def helper(tokens):
            token = tokens.pop(0)
            if token == '(':
                L = []
                while tokens[0] != ')':
                    L.append(helper(tokens))
                tokens.pop(0)
                return L
            else:
                return token
        return helper(tokens.copy())

    tree = parse_sexp(train_TOP)

    entities = []

    def extract_entities(tree, current_label=None, text_accumulator=[]):
        if isinstance(tree, list):
            label = tree[0]
            content = tree[1:]
            text = []
            for item in content:
                extract_entities(item, label, text)
            entity_text = ' '.join(text)
            if label in ['ORDER', 'PIZZAORDER', 'NOT'] or label not in ['NUMBER']:
                match = re.search(re.escape(entity_text), train_SRC)
                if match:
                    entities.append({
                        'label': label,
                        'word': match.group(),
                    })
            text_accumulator.extend(text)
        else:
            text_accumulator.append(tree)

    extract_entities(tree)

    result = {
        'sentence': train_SRC,
        'entities': entities
    }
    print(result)
    return result



if __name__ == '__main__':
    text = rtc('./test.json')
    toks = preprocess_train_top_decoupled(text)
    # for i in range(len(toks)):
    #     tree = linearized_rep_to_tree_rep(toks[i])
    #     for pre, fill, node in RenderTree(tree):
    #         print("%s%s" % (pre, node.id))
    #     tags = traverse_tree(tree)

    #     print(tags)

    result = parse_tc("","")
    
