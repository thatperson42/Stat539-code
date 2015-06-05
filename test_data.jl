#test_data.jl
using Base.Test

@test ndcg_loss([1:5], [1:5]) < 10.0^(-5)

for i in 1:100
    @test ndcg_loss(rand(10), rand(10)) <= 1.0
    @test ndcg_loss(rand(10), rand(10)) >= 0.0
    @test !isnan(ndcg_loss(rand(10), rand(10)))
end

