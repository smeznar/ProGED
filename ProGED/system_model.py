# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

"""Module implementing the SystemModel class that represents a system of equations 
defined by a vector of canonical expression strings.
    
An object of SystemModel acts as a container for various representations of the model,
including its vector of expressions, symbols, parameters, the parse trees that simplify to it,
and associated information and references. 
Class methods serve as an interfance to interact with the model.
The class is intended to be used as part of an equation discovery algorithm."""

class SystemModel:
    def __init__(self, expr, p=0, params=[], sym_params=[], sym_vars = [], info=None):
        """Initialize a SystemModel.
        
        Arguments:
            TODO
            """
        
        if not info:
            self.info = {}
        else:
            self.info = info
        
        if isinstance(expr, str):
            self.expr = [sp.sympify(ex) for ex in expr.strip(" ()").split(",")]
        else:
            self.expr = expr
     
        """sym_params should be tuple of tuples of sympy symbols"""
        self.sym_params = sym_params
        self.params = params
        self.n_params = [len(par) for par in sym_params]

        """sym_vars should be tuple of tuples of sympy symbols"""
        self.sym_vars = sym_vars

        self.p = p

        """TODO: figure out structure for trees"""
        #self.trees = {} #trees has form {"code":[p,n]}"
        #if len(code)>0:
            #self.add_tree(code, p)

        self.estimated = {}
        self.valid = False

    def set_estimated(self, result, valid=True):
        """Store results of parameter estimation and set validity of model according to input.
        
        Arguments:
            result (dict): Results of parameter estimation. 
                Designed for use with methods, implemented in scipy.optimize, but works with any method.
                Required items:
                    "x": solution of optimization, i.e. optimal parameter values (list of floats)
                    "fun": value of optimization function, i.e. error of model (float).
            valid: True if the parameter estimation succeeded.
                Set as False if the optimization was unsuccessfull or the model was found to not fit 
                the requirements. For example, we might want to limit ED to models with 5 or fewer parameters
                due to computational time concerns. In this case the parameter estimator would refuse
                to fit the parameters and set valid = False. 
                Invalid models are typically excluded from post-analysis."""
        
        self.estimated = result
        self.valid = valid
        if valid:
            self.params = np.split(result["x"], np.cumsum(self.n_params))

    def get_error(self, dummy=10**8):
        """Return model error if the model is valid, or dummy if the model is not valid.
        
        Arguments:
            dummy: Value to be returned if the parameter have not been estimated successfully.
            
        Returns:
            error of the model, as reported by set_estimated, or the dummy value.
        """
        if self.valid:
            return self.estimated["fun"]
        else:
            return dummy

    def set_params(self, params, split=True):
        if split:
            self.params = np.split(params, np.cumsum(self.n_params))
        else:
            self.params=params

    def full_expr (self, params=None):
        """Substitutes parameter symbols in the symbolic expression with given parameter values.
        
        Arguments:
            params (list of floats): Parameter values.
            
        Returns:
            sympy expression."""
        if not params:
            params = self.params

        fullexprs = []
        for i, ex in enumerate(self.expr):
            if type(self.sym_params[i]) != type((1,2)):
                fullexprs += [ex.subs([[self.sym_params[i], params[i]]])]
            else:
                fullexprs += [ex.subs(list(zip(self.sym_params[i], params[i])))]
                
        return fullexprs

    def lambdify (self, params=None, arg="numpy"):
        """Produce a callable function from the symbolic expression and the parameter values.
        
        This function is required for the evaluate function. It relies on sympy.lambdify, which in turn 
            relies on eval. This makes the function somewhat problematic and can sometimes produce unexpected
            results. Syntactic errors in variable or parameter names will likely produce an error here.
        
        Arguments:
            arg (string): Passed on to sympy.lambdify. Defines the engine for the mathematical operations,
                that the symbolic operations are transformed into. Default: numpy.
                See sympy documentation for details.
                
        Returns:
            callable function that takes variable values as inputs and return the model value.
        """
        if not params:
            params = self.params
        fullexprs = self.full_expr(params)
        lambdas = [sp.lambdify(self.sym_vars, full_expr, arg) for full_expr in fullexprs]
        return lambda x: np.transpose([lam(*x.T) for lam in lambdas]), lambdas


    def evaluate (self, points, *args):
        """Evaluate the model for given variable and parameter values.
        
        If possible, use this function when you want to do computations with the model.
        It relies on lambdify so it shares the same issues, but includes some safety checks.
        Example of use with stored parameter values:
            predictions = model.evaluate(X, *model.params)
        
        Arguments:
            points (numpy array): Input data, shaped N x M, where N is the number of samples and
                M the number of variables.
            args (list of floats): Parameter values.
            
        Returns:
            Numpy array of shape N x D, where N is the number of samples and D the number of output variables.
        """
        lamb_expr = sp.lambdify(self.sym_vars, self.full_expr(*args), "numpy")
        
        if type(points[0]) != type(np.array([1])):
            if type(lamb_expr(np.array([1,2,3]))) != type(np.array([1,2,3])):
                return np.ones(len(points))*lamb_expr(1)
            return lamb_expr(points)
        else:
#            if type(lamb_expr(np.array([np.array([1,2,3])]).T)) != type(np.array([1,2,3])):
            return lamb_expr(*points.T)

    def __str__(self):
        return str(self.expr)
    
    def __repr__(self):
        return str(self.expr)
    