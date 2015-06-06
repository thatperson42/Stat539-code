
using Winston
include("load_data.jl")

# takes 80 seconds to load in train.txt
@time queries_count, queries_dict, relevance, features_mat = load_ms_data("Fold1/train.txt");

#@time queries_count, queries_dict, relevance, features_mat = load_all_data();
queries = collect(keys(queries_dict));
normalize_features!(features_mat)
scores_dict_true = calc_asympt_scores(queries_dict, relevance);

lambda_val = 0.1
eta_val = 0.01
k_agg = 100
#n_iter = 10^4 # For testing
n_iter = 2*10^5
k_vals = [1, 10, 100, 1000, 10000]
#k_vals = [1, 100]
n_experiments = 10

@time thetas = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat);

@time thetas, risk_tally = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat, "leastsquares", true);
@time thetas = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat, "leastsquares_true");
@profile thetas = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat, "leastsquares_true");
ProfileView.view()
@time thetas = run_updater(lambda_val, eta_val, k_agg, n_iter, queries_dict, relevance, features_mat, "surrogate_pair");


@time risk_est = estimate_ndcg_loss(queries, queries_dict, scores_dict_true, features_mat,
                            thetas[:,1000], 50)


# todo: start storing partway information
function run_test(k_vals, n, lambda_val, eta_val,
                  queries_dict, relevance, features_mat, est_loss=false)
    
# pairwise surrogate loss
    println("At pairwise surrogate loss")
    @time thetas_pairwise = run_updater(lambda_val, eta_val, 0, n, queries_dict, 
                         relevance, features_mat, "surrogate_pair")

# aggregated surrogate NDCG
    thetas_ndcg = Dict()
    risk_est = Dict()
    for k in k_vals
        #n_tmp = int(ceil(n / k))
        n_tmp = n
        println("At NDCG surrogate loss, aggregation ", k)
        if est_loss
            @time thetas_ndcg[k], risk_est[k] = run_updater(lambda_val, eta_val, k,
                                         n_tmp, queries_dict, 
                                         relevance, features_mat, "leastsquares", est_loss)
        else
            @time thetas_ndcg[k] = run_updater(lambda_val, eta_val, k, n_tmp, queries_dict, 
                                         relevance, features_mat)
        end
    end

#=
# todo: make sure the number of iterations is right. It isn't now
    thetas_score = run_updater(lambda_val, eta_val, 0, n_iter, 
                          queries_dict, relevance, features_mat, "leastsquares_true");
=#
    if est_loss
        return thetas_pairwise, thetas_ndcg, risk_est
    else
        return thetas_pairwise, thetas_ndcg
    end
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

eta_vals = log10(logspace(0.001, 10.0, n_experiments))
    eta_val = eta_vals[i]
risks_plot_tmp = Dict()

risk_pairwise = zeros(n_experiments)
k_vals = [10, 100]
risk_ndcg = zeros(n_experiments, length(k_vals))
risks_ndcg_partway = Dict()
for i in 1:n_experiments

    println("---- iteration ", i, " ----")
    tic()
    @time thetas_pairwise, thetas_ndcg, risks_ndcg_partway[i] = run_test(k_vals, n_iter, lambda_val, eta_val,
                      queries_dict, relevance, features_mat, true);

    println(thetas_pairwise[:, end])
    println(thetas_ndcg[k_vals[1]][:, end])
    println(thetas_ndcg[k_vals[end]][:, end])
    @time risk_pairwise[i] = estimate_ndcg_loss(queries, queries_dict, scores_dict_true,
                                                features_mat, thetas_pairwise[:, end])
    for k_ind in 1:length(k_vals)
        k = k_vals[k_ind]
        println("Evaluating risk for ", k)
        @time risk_ndcg[i, k_ind] = estimate_ndcg_loss(queries, queries_dict, scores_dict_true,
                  features_mat, thetas_ndcg[k][:, end])
    end

    @time thetas_ndcg_true = run_updater(lambda_val, eta_val, 0, n_iter, queries_dict, 
                                     relevance, features_mat, "leastsquares_true")
    @time risk_ndcg_true[i] = estimate_ndcg_loss(queries, queries_dict, scores_dict_true,
                                                features_mat, thetas_ndcg_true[:, end])
    toc()
    println("-------------------")
end

println(risk_pairwise)
mean(risk_pairwise)
std(risk_pairwise)
[mean(risk_ndcg[:,k_ind]) for k_ind in 1:length(k_vals)]
[std(risk_ndcg[:,k_ind]) for k_ind in 1:length(k_vals)]
println(risk_ndcg)
means_pairwise = mean(risk_pairwise)
#0.6558810
std_pairwise = std(risk_pairwise) / sqrt(n_experiments)
#0.00155878
means_ndcg_true = mean(risk_ndcg_true)
#0.6692532482415287
std_ndcg_true = std(risk_ndcg_true) / sqrt(n_experiments)
#9.573520721318257e-5
means_ndcg = mapslices(mean, risk_ndcg, 1)
#0.670931  0.667591  0.663506  0.663353
std_ndcg = mapslices((x) -> std(x) / sqrt(n_experiments), risk_ndcg, 1)
#0.0015444  0.00114515  0.000955365  0.000673178

w = length(k_vals)
p = semilogx(k_vals, means_ndcg)
p1 = Curve(k_vals, means_ndcg - 1.96*std_ndcg, color="blue")
p2 = Curve(k_vals, means_ndcg + 1.96*std_ndcg, color="blue")
q1 = Curve(k_vals, [means_ndcg_true for i in 1:w])
q2 = Curve(k_vals, [means_ndcg_true - 1.96*std_ndcg_true for i in 1:w], color="magenta")
q3 = Curve(k_vals, [means_ndcg_true + 1.96*std_ndcg_true for i in 1:w], color="magenta")

c1 = Curve(k_vals, [mean(risk_pairwise) for i in 1:w])
c2 = Curve(k_vals, [means_pairwise - 1.96*std_pairwise for i in 1:w], color="red")
c3 = Curve(k_vals, [means_pairwise + 1.96*std_pairwise for i in 1:w], color="red")
add(p, p1)
add(p, p2)
add(p, q1)
add(p, q2)
add(p, q3)
add(p, c1)
add(p, c2)
add(p, c3)
savefig("Figure3.png")

risks_ndcg_partway_tmp = risks_ndcg_partway
mean_risk = zeros(length(risks_ndcg_partway[1][10]), length(k_vals))
std_risk = zeros(length(risks_ndcg_partway[1][10]), length(k_vals))
for k_ind in 1:length(k_vals)
    k = k_vals[k_ind]
    for j in 1:length(risks_ndcg_partway[1][k])
        vals = [risks_ndcg_partway[i][k][j] for i in 1:n_experiments]
        mean_risk[j, k_ind] = mean(float64(vals))
        std_risk[j, k_ind] = std(float64(vals))
    end
end

x_vals = (1:(n_iter / 100)) * 100
for k_ind in 1:length(k_vals)
    p = plot(x_vals, mean_risk[:,k_ind], xlabel="Iteration",
             ylabel="Estimated risk", "-")
    title(string("k=", k_vals[k_ind]))
    c1 = Curve(x_vals, mean_risk[:,k_ind] - 1.6*std_risk[:,k_ind]/sqrt(n_experiments), color="red")
    c2 = Curve(x_vals, mean_risk[:,k_ind] + 1.6*std_risk[:,k_ind]/sqrt(n_experiments), color="red")
    add(p, c1)
    add(p, c2)
    savefig(string("Estim_risk_k", k_vals[k_ind], ".png"))
end


