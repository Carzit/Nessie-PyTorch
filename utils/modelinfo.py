from typing import Any, List, Callable

class ModelInfo:
    def __init__(self, name:str, alias:List[str], net_class:Callable, q_class:Callable, **kwargs) -> None:
        self.name = name
        self.alias = alias
        self.net_class = net_class 
        self.q_class = q_class
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def __eq__(self, value: str) -> bool:
        return self.name == value or value in self.alias
    
    def __repr__(self):
        return f"ModelInfo(name={self.name}, alias={str(self.alias)})"
    
class ModelInfoList:
    def __init__(self, modelinfo_list:List[ModelInfo]) -> None:
        self.modelinfo_list = modelinfo_list
    
    def __len__(self):
        return len(self.modelinfo_list)

    def __getitem__(self, index):
        if isinstance(index, str):
            results = []
            for i in self.modelinfo_list:
                if i == index:
                    results.append(i)
            if len(results) == 1:
                results = results[0]
        if isinstance(index, int):
            results = self.modelinfo_list[index]
        if isinstance(index, slice):
            results = self.modelinfo_list[index]
        return results
    
    def __repr__(self):
        return f"ModelInfoList{str([modelinfo.name for modelinfo in self.modelinfo_list])}"
    