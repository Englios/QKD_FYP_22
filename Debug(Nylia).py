def gibo(n,m,a,b):
    #gn=gn-1+gn-2
    if n == 0:
        return 1
    
    if n == 1:
        return 0
        
    if n == m:
        return a
    
    if n == m+1:
        return b
    
    if n<m:
        return ((-1)**(-n+1))*(gibo(abs(n)-1,m,a,b)-gibo(abs(n)-2,m,a,b))
    
    else:
        return gibo(n-1,m,a,b)-gibo(n-2,m,a,b)
        
print(gibo(-4,3,1,2))

# def f(n):
#         if n == 0:
#             return 0
#         if n == 1:
#             return 1
#         if n < 0:
#             return ((-1)**(-n+1)) * f(-n)
#         return f(n-1)+f(n-2)
    
# print(f(-4))