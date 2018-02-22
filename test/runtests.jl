using Base.Test
using FEMHomogenization: remove_duplicates!

@test remove_duplicates!([]) == []
@test remove_duplicates!([3]) == [3]
@test remove_duplicates!([3,4]) == [3,4]
@test remove_duplicates!([3,3]) == [3]
@test remove_duplicates!([3,3,3]) == [3]
@test remove_duplicates!([3,4,5]) == [3,4,5]
@test remove_duplicates!([3,3,4]) == [3,4]
@test remove_duplicates!([3,4,4]) == [3,4]
@test remove_duplicates!([1,2,2,3,3,4,4]) == [1,2,3,4]