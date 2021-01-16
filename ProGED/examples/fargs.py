def f(a,b, c=300, d=23, e=34,f=34):
    print(a,b,c,d,e,f)
    
    li = [a,b,c,d,e,f]
    return a+b
print(f(1,2, d=4, f=6,e=5))
def g(a,d, k=12, *lil):
    # print(a,d,k,*lil)
    f(*lil)
    # print(f(4, 5, 6))
    # print(f(lil))
    # print(*lil)
    return 101
# print(g(1,2,c=12,f=445, 4))
print(" --- g --- ")
g(1, 2, 3, 4, 5, 6) #, 4))
# print(())
# print("empty")

