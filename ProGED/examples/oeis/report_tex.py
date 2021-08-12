inputa = """
error: 3.203874099327748e-07   model: 1.999*an_1 - 0.999*an_2 + 1.000*n - 0.000
error: 0.08057339090814337     model: 0.999*an_1 + 1.182*sqrt(n) + 0.514*n**2
error: 9.016581762817749       model: 1.579*an_1 - 0.586*an_3 + 4.0                   
error: 9.318342794960778       model: 2.122*an_1 - 1.128*an_2                         
"""


import re
line = "error: 3.203874099327748e-07   model: 1.999*an_1 - 0.999*an_2 + 1.000*n - 0.000"
def line2ara(line):

    def expo(exp):
        expb = f"* 10^{{{int(exp[2:])}}}"  
        return expb
    # print(expo('e-07'))
    # 1/0
    # def expall(line):
    #     exps = re.findall('e[+-]\d{1,2}', line)
    #     newline = line
    #     for exp in exps:
    #         expb = f"* 10^{{{int(exp[2:])}}}"  
    #         newline = re.sub(exp, expb, newline)
    #     return newline
    # print(expo(line))
    def resub(line: str, restr: str, frepl):
        founds = re.findall(restr, line)
        # print('founds in line:', line, founds)
        newline = line
        for found in founds:
            fond = "".join([f"[{i}]" for i in found])
            newline = re.sub(fond, frepl(found), newline)
        return newline
    line = resub(line, 'e[+-]\d{1,2}', lambda exp: f"* 10^{{{int(exp[1:])}}}")
    line = resub(line, 'an_\d{1,2}', lambda an: f"a_{{n-{int(an[3:])}}}")
    line = resub(line, '\*\*\d{1,2}', lambda power: f"^{power[2:]}")

    #  flip sides:
    error = re.findall('error:(.+)model:(.+)', line)
    line = error[0][1] + ' & ' + error[0][0] + '\\\\'

    # def an(an_1):
    #     an_1 = re.findall('an_\d{1,2}', an_1)
    #     a_n1 = f"a_{{n-{int(an_1[0][3:])}}}"  
    #     return a_n1
    # # print(an(line))
    # sti = re.sub('model:', '', re.sub('error:', '', line))
    return line
# arline(line)
# print(line2ara(line))

lines = re.findall('(.+)\n', inputa)
for line in lines:
    # print(line)
    print(line2ara(line))


# linea = lines[1]
# print(line2ara(linea))

# print(linea)
# fond = "".join([f"[{i}]" for i in "**2"])
# print(fond)
# print(re.findall('aa\d{1,2}', 'aaa1'))
# print(re.findall(fond, '***2'))
# newline = re.sub(r"[*][2]", f"^2", '****2')
# print(newline)
