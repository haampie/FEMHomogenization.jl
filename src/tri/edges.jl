
struct Edge{Ti}
    from::Ti
    to::Ti
end

"""
Fancy Edge constructor
"""
a → b = Edge(a, b)

"""
Store an edge uniquely
"""
↔(a, b) = a < b ? a → b : b → a