#test_data.jl
# To be run in REPL
include("load_data.jl")
using Base.Test

@test ndcg_loss([1:5], [1:5]) < 10.0^(-5)

for i in 1:100
    @test ndcg_loss(rand(10), rand(10)) <= 1.0
    @test ndcg_loss(rand(10), rand(10)) >= 0.0
    @test !isnan(ndcg_loss(rand(10), rand(10)))
end

# TODO: test to make sure that k-aggregation is approximating the asymptotic scores
scores_dict_true = calc_asympt_scores(queries_dict, relevance);
# todo: sample some query q
q = collect(keys(queries_dict))[1];
scores = scores_dict_true[q];
n_agg_sizes = 7;
s_est = zeros(length(scores), n_agg_sizes);
est_diff = zeros(n_agg_sizes)
for (k_ind in 1:n_agg_sizes)
    k = 10^k_ind;
    s_est[:,k_ind] = estimate_scores(relevance, queries_dict, q, k);
    est_diff[k_ind] = norm(scores - s_est[:,k_ind]);
end
println(est_diff)
#[9.05194840518518,9.014597206595749,8.507949087832257,5.5844844637738085,1.5182813940889868,0.17995898366133461,0.05503376514770855]

