import sys
import json

aStepsDefault = 0;
mStepsDefault = 0;

class PTuple:
    def __init__(self,data = []):
        """Constructor"""
        self.parameters = data
        
    def __mul__(self, p):
        params = self.parameters.copy()
        params.append(p)
        pt = PTuple(params)
        return pt

def cartesianProduct(A,B):
    res = []
    if len(A) == 0:
        for b in B:
            res.append(PTuple([b]))
        return res;
    for a in A:
        for b in B:
            res.append(a*b)
    return res

def Tuple(params):
    points = []
    for bank  in params:
        points = cartesianProduct(points, bank)
    return points

def fillParameterList(p, parameterList, errorsList = []):
    if isinstance(p, int) or isinstance(p, float):
        parameterList.append((p))
        return
    if isinstance(p, list):
        for el in p:
            fillParameterList(el,parameterList,errorsList)
        return
    if isinstance(p, dict):
        #проверки
        if not "begin" in p:
            errorsList.append("ошибка диапазона: отсутствует поле begin")
            return
        if not "step" in p:
            errorsList.append("ошибка диапазона: отсутствует поле step")
            return
        if not "end" in p:
            errorsList.append("ошибка диапазона: отсутствует поле end")
            return
        if not (isinstance(p["begin"], int) or isinstance(p["begin"], float)):
            errorsList.append("ошибка диапазона: поле begin должно иметь численное значение")
            return
        if not (isinstance(p["end"], int) or isinstance(p["end"], float)):
            errorsList.append("ошибка диапазона: поле end должно иметь численное значение")
            return
        if not (isinstance(p["step"], int) or isinstance(p["step"], float)):
            errorsList.append("ошибка диапазона: поле step должно иметь численное значение")
            return
        val = begin = p["begin"]
        end = p["end"]
        step = p["step"]
        
        
        if begin < end:
            while val < end:
                parameterList.append((round(val,2)))
                val += abs(step)
        if begin > end:
            while val > end:
                parameterList.append((round(val,2)))
                val -= abs(step)
        parameterList.append((end))
        return
    
    errorsList.append("ошибка формата: неподдерживаемый тип параметра:" + str(type(p)))

def return_task_list(fname):
    with open(fname) as json_file:
        jsn = json.load(json_file)
    
    tpl = jsn["tpl"]
    for task in jsn["tasks"]:
        data = []
        errors = []
        # XXX: This trush badly needs refactorin
            
        for p in tpl:
            plist = []
            fillParameterList(task[p],plist,errors)
            if len(errors) > 0:
                for err in errors:
                    print(err)
                exit()
            data.append(plist)
        
    
        points = Tuple(data)
        res = []
        for point in points:
            res.append(point.parameters)
        return res
            
