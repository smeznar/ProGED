# -*- coding: utf-8 -*-

import numpy as np
import sympy.core as sp
from sympy.simplify import simplify as sympy_simplify
from sympy import symbols as sympy_symbols
import pickle

from ProGED.model import Model

"""Implements ModelBox, an class that stores and manages a collection of Model instances."""

class ModelBox:
    """Container for a dictionary of Model instances, as well as accompanying methods.
    
    Object that stores and organizes Model instances in a dictionary. 
    The keys are canonical expression strings and the values are Model instances.
    Newly generated expressions are processed, simplified into its canonical respresentation,
    and stored into the appropriate dictionary item. 
    The string representation of ModelBox produces a nice printout of the contents.
    
    Attributes:
        models_dict (dict): Dictionary with canonical expression strings as keys 
            and Model instances as values.
    
    Methods:
        add_model: Proccess and store a new model.
        verify_expression: Used by add_model. Check if expression fits basic requirements.
        enunerate_constants: Used by add_model. Identifies and enumerates parameter symbols in expression.
        simplify_constants: Used by add_model. Attemps to simplify expression by considering free constants.
        string_to_canonic_expression: Used by add_model. Series of transformation that
            produce a canonic representation.
        keys: Return the canonical expression strings acting as keys for models_dict.
        values: Return the Model instances in models_dict.
        items: Return the key and value pairs in models_dict.
    """
    def __init__ (self, models_dict = {}):
        self.models_dict = dict(models_dict)
        
    def add_model (self, expr_str, symbols, grammar, code="0", p=1.0, **kwargs):
        x = [s.strip("'") for s in symbols["x"]]
        expr, symbols_params = self.string_to_canonic_expression(expr_str, symbols)
        
        if not self.verify_expression(expr, symbols=symbols, grammar=grammar, code=code, 
                                      p=p, x=x, symbols_params=symbols_params):
            return False, expr
        
        if str(expr) in self.models_dict:
            self.models_dict[str(expr)].add_tree(code, p)
        else:
            if "params" in kwargs:
                params = kwargs["params"]
            else:
                params = [(np.random.random()-0.5)*10 for i in range(len(symbols_params))]
                
            self.models_dict[str(expr)] = Model(grammar=grammar, expr = expr, sym_vars = x, sym_params = symbols_params, params=params, code=code, p=p)
        
        return True, str(expr)
    
    def verify_expression(self, expr, **kwargs):
        good = False
        """Sanity test + check for complex infinity"""
        if len(kwargs["symbols_params"]) > -1 and not "zoo" in str(expr):
            """Check if expr contains at least one variable"""
            for xi in kwargs["x"]:
                if xi in str(expr):
                    good = True
                    break
        return good
    
    def enumerate_constants(self, expr, symbols):
        """ Enumerates the constants in a Sympy expression. 
        Example: C*x**2 + C*x + C -> C0*x**2 + C1*x + C2
        Input:
            expr - Sympy expression object
            symbols - dict of symbols used in the grammar. Required keys: "const", which must be a single character, such as "'C'".
        Returns:
            Sympy object with enumerated constants
            list of enumerated constants"""
        if isinstance(symbols["const"], list):
            csym = symbols["const"]
        elif isinstance(symbols["const"], str):
            csym = [symbols["const"]]
            
        poly2 = np.array(list(str(expr)), dtype='<U16')
        for c in csym:
            constind = np.where(poly2 == symbols["const"].strip("'"))[0]
            """ Rename all constants: c -> cn, where n is the power of the associated term"""
            constants = [symbols["const"].strip("'")+str(i) for i in range(len(constind))]
            poly2[constind] = constants
        """ Return the sympy object """
        return sp.sympify("".join(poly2)), tuple(constants)
    
    def enumerate_constants_2(self, expr, symbols):
        """ Enumerates the constants in a Sympy expression. 
        Example: C*x**2 + C*x + C -> C0*x**2 + C1*x + C2
        Input:
            expr - Sympy expression object
            symbols - dict of symbols used in the grammar. Required keys: "const", which must be a single character, such as "'C'".
        Returns:
            Sympy object with enumerated constants
            list of enumerated constants"""
        if isinstance(symbols["const"], list):
            csym = symbols["const"]
        elif isinstance(symbols["const"], str):
            csym = [symbols["const"]]
            
        poly2 = np.array(list(str(expr)), dtype='<U16')
        for c in csym:
            constind = np.where(poly2 == symbols["const"].strip("'"))[0]
            """ Rename all constants: c -> cn, where n is the power of the associated term"""
            constants = [symbols["const"].strip("'")+str(i) for i in range(len(constind))]
            poly2[constind] = constants

        return "".join(poly2), tuple(constants)
    
    def simplify_constants_2 (self, eq, c, var):
        if len(eq.args) == 0:
            return eq in var, eq is c, [(eq,eq)]
        else:
            has_var, has_c, subs = [], [], []
            for a in eq.args:
                a_rec = self.simplify_constants (a, c, var)
                has_var += [a_rec[0]]; has_c += [a_rec[1]]; subs += [a_rec[2]]
            if sum(has_var) == 0 and True in has_c:
                return False, True, [(eq, c)]
            else:          
                args = []
                if isinstance(eq, (sp.add.Add, sp.mul.Mul)):
                    has_free_c = False
                    if True in [has_c[i] and not has_var[i] for i in range(len(has_c))]:
                        has_free_c = True
                        
                    for i in range(len(has_var)):
                        if has_var[i] or (not has_free_c and not has_c[i]):
                            if len(subs[i]) > 0:
                                args += [eq.args[i].subs(subs[i])]
                            else:
                                args += [eq.args[i]]
                    if has_free_c:
                        args += [c]
                    
                else:
                    for i in range(len(has_var)):
                        if len(subs[i]) > 0:
                            args += [eq.args[i].subs(subs[i])]
                        else:
                            args += [eq.args[i]]
                return True in has_var, True in has_c, [(eq, eq.func(*args))]
            
    def simplify_constants (self, eq, c, var):
        if len(eq.args) == 0:
            if eq in var:
                return True, False, [(eq, eq)]
            elif str(eq)[0] == str(c):
                return False, True, [(eq, eq)]
            else:
                return False, False, [(eq, eq)]
        else:
            has_var, has_c, subs = [], [], []
            for a in eq.args:
                a_rec = self.simplify_constants (a, c, var)
                has_var += [a_rec[0]]; has_c += [a_rec[1]]; subs += [a_rec[2]]
            if sum(has_var) == 0 and True in has_c:
                return False, True, [(eq, c)]
            else:          
                args = []
                if isinstance(eq, (sp.add.Add, sp.mul.Mul)):
                    has_free_c = False
                    if True in [has_c[i] and not has_var[i] for i in range(len(has_c))]:
                        has_free_c = True
                        
                    for i in range(len(has_var)):
                        if has_var[i] or (not has_free_c and not has_c[i]):
                            if len(subs[i]) > 0:
                                args += [eq.args[i].subs(subs[i])]
                            else:
                                args += [eq.args[i]]
                    if has_free_c:
                        args += [c]
                    
                else:
                    for i in range(len(has_var)):
                        if len(subs[i]) > 0:
                            args += [eq.args[i].subs(subs[i])]
                        else:
                            args += [eq.args[i]]
                return True in has_var, True in has_c, [(eq, eq.func(*args))]
            
    def string_to_canonic_expression_2 (self, expr_str, symbols={"x":["'x'"], "const":"'C'", "start":"S", "A":"A"}):
        """Convert the string into the canonical Sympy expression.

        Input:
            expr_str -- String expression e.g. joined sample string
                generated from grammar.
        Output:
            expr -- Sympy expression object in canonical form.
            symbols_params -- Tuple of enumerated constants.
        """
        x = [sympy_symbols(s.strip("'")) for s in symbols["x"]]
        c = sympy_symbols(symbols["const"].strip("'"))
        expr = sp.sympify(expr_str)
        expr = self.simplify_constants(expr, c, x)[2][0][1]
        expr, symbols_params = self.enumerate_constants(expr, symbols)
        return expr, symbols_params
    
    def string_to_canonic_expression (self, expr_str, symbols={"x":["'x'"], "const":"'C'", "start":"S", "A":"A"}):
        """Convert the string into the canonical Sympy expression.

        Input:
            expr_str -- String expression e.g. joined sample string
                generated from grammar.
        Output:
            expr -- Sympy expression object in canonical form.
            symbols_params -- Tuple of enumerated constants.
        """
        x = [sympy_symbols(s.strip("'")) for s in symbols["x"]]
        c = sympy_symbols(symbols["const"].strip("'"))
        
        expr = self.enumerate_constants(expr_str, symbols)[0]
        expr = self.simplify_constants(expr, c, x)[2][0][1]
        expr = self.enumerate_constants(expr, symbols)[0]
        expr = self.simplify_constants(expr, c, x)[2][0][1]
        
        expr, symbols_params = self.enumerate_constants(expr, symbols)
        return expr, symbols_params
    
    def retrieve_best_models (self, N = 3):
        """Returns the top N models, according to their error.
        
        Arguments:
            N (int): The number of models to return. Default: 3.
            
        Returns:
            ModelBox containing only the top N models.
        """
        
        models_keys = list(self.models_dict.keys())
        errors = [self.models_dict[m].get_error() for m in self.models_dict]
        sortind = np.argsort(errors)
        models2 = [self.__getitem__(int(n)) for n in sortind[:N]]
        keys2 = [models_keys[n] for n in sortind[:N]]
        
        return ModelBox(dict(zip(keys2, models2)))
    
    def split (self, n_batches = None, batch_size = None):
        """Splits the ModelBox into a number of batches, each also a ModelBox.
        Either the number of batcher or the batch size must be provided. 
        If both are given, batch size is ignored.
        
        The batching is performed by convering the dictionary into a list of tuples,
        following default ordering. The list is split into sublists, which are converted
        back into dictionaries and into ModelBoxes. 

        Parameters
        ----------
        n_batches (int): The number of batches to create. 
        batch_size (int): The size of each batch. Ignored if n_batches is not None.

        Returns
        -------
        List of ModelBox instances.
        """
        if not n_batches and not batch_size:
            raise ValueError("ModelBox.split requires either n_batches or batch_size argument.")
        elif not batch_size:
            batch_size = int(np.ceil(len(self) / n_batches))
        elif not n_batches:
            n_batches = int(np.ceil(len(self) / batch_size))
            
        return [ModelBox(dict(list(self.models_dict.items())[i*batch_size : min([(i+1)*batch_size, len(self.models_dict)])])) for i in range(min([n_batches, len(self.models_dict)]))]
        
    def dump (self, file_path):
        """Stores the ModelBox to a file by invoking pickle.
        
        Parameters
        ----------
        file_path (string): The path and name of the file to be created.
        
        Returns
        -------
        None
        """
        with open(file_path, "wb") as file:
                pickle.dump(self.models_dict, file)
                
    def load (self, file_path):
        """Loads the contents of a ModelBox from a file, created by ModelBox.dump.
        The loaded models are merged into the existing dictionary. In the case of clashes,
        the loaded values override the existing ones. 
        Invokes pickle.load.
        
        Parameters
        ----------
        file_path (string): The path and name of the file to load from.
        
        Returns
        -------
        None
        """
        with open(file_path, "rb") as file:
            loaded_models = pickle.load(file)
            
        self.models_dict.update(loaded_models)
                

    def __str__(self):
        txt = "ModelBox: " + str(len(self.models_dict)) + " models"
        for m in self.models_dict:
            txt += "\n-> " + str(self.models_dict[m].expr) + ", p = " + str(self.models_dict[m].p)
            txt += ", parse trees = " + str(len(self.models_dict[m].trees))
            txt += ", valid = " + str(self.models_dict[m].valid)
            if self.models_dict[m].valid:
                txt += ", error = " + str(self.models_dict[m].get_error())
        return txt
    
    def keys(self):
        return self.models_dict.keys()
    
    def values(self):
        return self.models_dict.values()
    
    def items(self):
        return self.models_dict.items()
    
    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self.models_dict)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.models_dict[key]
        elif isinstance(key, int):
            return list(self.models_dict.values())[key]
        else:
            raise KeyError ("Invalid key for model dictionary. "\
                            "Expected canonical expression string or integer index.")
        
def symbolic_difference(ex1, ex2, thr = 9):
    dif = sp.N(sympy_simplify(ex2 - ex1))
    for a in sp.preorder_traversal(dif):
        if isinstance(a, sp.Float):
            dif = dif.subs(a, round(a, thr))
    return sympy_simplify(dif)

if __name__ == "__main__":
    from nltk import PCFG
    print("--- models_box.py test ---")
    grammar_str = "S -> 'c' '*' 'x' [0.5] | 'x' [0.5]"
    grammar = PCFG.fromstring(grammar_str)
    expr1_str = "x"
    expr2_str = "c*x"
    symbols = {"x":['x'], "const":"c", "start":"S"}
    
    models = ModelBox()
    print(models.add_model(expr1_str, symbols, grammar))
    print(models.add_model(expr2_str, symbols, grammar, p=0.5, code="1"))
    
    print(models)
    