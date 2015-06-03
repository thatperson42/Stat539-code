
include("load_data.jl")

# takes 80 seconds to load in train.txt
@time queries_count, queries_dict, relevance, features_mat = load_ms_data("Fold1/train.txt");
#@time queries_count, queries_dict, relevance, features_mat = load_all_data();
queries = collect(keys(queries_dict));
normalize_features!(features_mat)
scores_dict_true = calc_asympt_scores(queries_dict, relevance);

lambda_val = 0.1
eta_val = 100
k_agg = 100
#n_iter = 10^4
n_iter = 2*10^5
#k_vals = [1, 10, 100, 1000]
k_vals = [1, 100]

@time thetas = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat);

@time thetas = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat, "leastsquares_true");
@profile thetas = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat, "leastsquares_true");
ProfileView.view()
@time thetas = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat, "surrogate_pair");


@time risk_est = estimate_ndcg_loss(queries, queries_dict, scores_dict_true, features_mat,
                            thetas[:,1000], 50)


# todo: start storing partway information
function run_test(k_vals, n, lambda_val, eta_val,
                  queries_dict, relevance, features_mat)
    
# pairwise surrogate loss
    println("At pairwise surrogate loss")
    @time thetas_pairwise = run_updater(lambda_val, eta_val, 0, n, queries_dict, 
                         relevance, features_mat, "surrogate_pair")

# aggregated surrogate NDCG
    thetas_ndcg = Dict()
    for k in k_vals
        #n_tmp = int(ceil(n / k))
        n_tmp = n
        println("At NDCG surrogate loss, aggregation ", k)
        @time thetas_ndcg[k] = run_updater(lambda_val, eta_val, k, n_tmp, queries_dict, 
                                     relevance, features_mat)
    end

#=
# todo: make sure the number of iterations is right. It isn't now
    thetas_score = run_updater(lambda_val, eta_val, 0, n_iter, 
                          queries_dict, relevance, features_mat, "leastsquares_true");
=#
    return thetas_pairwise, thetas_ndcg
end

# TODO: why is aggregation 1 so much slower than pairwise??
# 4.2 seconds versus 218 seconds for aggregation 1, 232 for aggregation 10

@time thetas_pairwise, thetas_ndcg = run_test(k_vals, n_iter, lambda_val, eta_val,
                  queries_dict, relevance, features_mat);

#=
risk_pairwise = estimate_ndcg_loss(queries, queries_dict, scores_dict_true, features_mat,
                            thetas_pairwise[:, end])
risk_ndcg = Dict()
for k in k_vals
    risk_ndcg[k] = estimate_ndcg_loss(queries, queries_dict, scores_dict_true, features_mat,
                                thetas_ndcg[k][:, end])
end
risk_ndcg
=#

n_experiments = 1
risk_pairwise = zeros(n_experiments)
risk_ndcg = zeros(n_experiments, length(k_vals))
for i in 1:n_experiments
    println("---- iteration ", i, " ----")
    tic()
    @time thetas_pairwise, thetas_ndcg = run_test(k_vals, n_iter, lambda_val, eta_val,
                      queries_dict, relevance, features_mat);

    println(thetas_pairwise[:, end])
    println(thetas_ndcg[1][:, end])
    println(thetas_ndcg[10][:, end])
    @time risk_pairwise[i] = estimate_ndcg_loss(queries, queries_dict, scores_dict_true,
                                                features_mat, thetas_pairwise[:, end])
    for k_ind in 1:length(k_vals)
        k = k_vals[k_ind]
        println("Evaluating risk for ", k)
        @time risk_ndcg[i, k_ind] = estimate_ndcg_loss(queries, queries_dict, scores_dict_true,
                  features_mat, thetas_ndcg[k][:, end])
    end

    toc()
    println("-------------------")
end

println(risk_pairwise)
println(risk_ndcg)

risk_pairwise
0.685094
 0.655094
 0.676073
 0.666401
 0.657391
 0.67671 
 0.655747
 0.655865
 0.656398
 0.647723
# gives 0.6632 w/ std dev 0.012
risk_ndcg
Dict{Any,Any} with 10 entries:
  7  => {100=>0.6546883492186939,10=>0.6613630076106869,1000=>0.6622487385951138,1=>0.650121541447…
  4  => {100=>0.6558635683436541,10=>0.6575280298636348,1000=>0.6767600594855862,1=>0.677388184907…
  9  => {100=>0.6569437387647521,10=>0.6593249591937944,1000=>0.6739926167172102,1=>0.656218054257…
  10 => {100=>0.6524349795518835,10=>0.6812047059754145,1000=>0.6706352224886658,1=>0.670486066327…
  2  => {100=>0.6524087980921967,10=>0.6736081617778403,1000=>0.6706620228266958,1=>0.665053071016…
  3  => {100=>0.657857439811683,10=>0.6608097917862092,1000=>0.6447714413260504,1=>0.6418471980840…
  5  => {100=>0.6372417178548101,10=>0.669765251409642,1000=>0.6544477411420798,1=>0.6708440678773…
  8  => {100=>0.6676817709121099,10=>0.6824331782631172,1000=>0.6416694694802426,1=>0.692760568497…
  6  => {100=>0.655888369337893,10=>0.6531263507398304,1000=>0.6608256137280748,1=>0.6714356166798…
  1  => {100=>0.6882024660114637,10=>0.6722127192686054,1000=>0.651231416455179,1=>0.6539085581500…


for k in k_vals
    ww_tmp = zeros(10)
    for i in 1:10
        ww_tmp[i] = risk_ndcg[i][k]
    end
    println("for k=", k, ": mean=", mean(ww_tmp), ", sd=", std(ww_tmp))
end
for k=1: mean=0.665006292724529, sd=0.014859898844731309
for k=10: mean=0.6671376155888774, sd=0.01016230024607625
for k=100: mean=0.6579211197899141, sd=0.012996268933176122
for k=1000: mean=0.6607244342244898, sd=0.012381002484838095




