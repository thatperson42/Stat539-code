# load_data.jl
using DataArrays, DataFrames
using Distributions
using ProfileView

const d = 136 # number of dimensions

function load_ms_data(data_f)
    println("Will open data now")
    f = open(data_f)
    n_lines = countlines(data_f)
    features_mat = float32(zeros(d, n_lines))
    queries = zeros(n_lines)
    relevance = float32(zeros(n_lines))
    line_num = 0
# queries_dict will have queries as keys, array of line numbers as value
    queries_dict = Dict{Int, Array{Int, 1}}()
    queries_count = Dict{Int, Int}()
    for line in eachline(f)
        line_num += 1
        line_args = split(line)
        features = map((x) -> float(split(x,':')[2]), line_args[3:end])
# features_mat is the matrix of features
        features_mat[:, line_num] = features
        query_tmp = int(split(line_args[2], ':')[2])
        queries[line_num] = query_tmp
# relevance is the vector of relevance scores
        relevance[line_num] = int(line_args[1])

        if haskey(queries_dict, query_tmp)
            queries_count[query_tmp] += 1
            append!(queries_dict[query_tmp], [line_num])
        else
            queries_count[query_tmp] = 1
            queries_dict[query_tmp] = [line_num]
        end
    end
    close(f)

    @assert !any(isnan(relevance))
    @assert !any(isnan(features_mat))
    return queries_count, queries_dict, relevance, features_mat
end

function load_all_data()
    folders = map((x) -> string("Fold", x, "/"), 1:5)
    files = ["test.txt", "train.txt", "vali.txt"]
    data_fs = [string(folder, f) for folder in folders, f in files]

    println("Starting first file")
    queries_count, queries_dict, relevance, features_mat = load_ms_data(data_fs[1])
    for data_f in data_fs[2:end]
        println("Starting file ", data_f)
        queries_count_tmp, queries_dict_tmp, relevance_tmp, features_mat_tmp = load_ms_data(data_f)
        for query in keys(queries_count_tmp)
            if haskey(queries_dict, query)
                queries_count[query] += queries_count_tmp[query]
                append!(queries_dict[query], queries_dict_tmp[query])
            else
                queries_count[query] = queries_count_tmp[query]
                queries_dict[query] = queries_dict_tmp[query]
            end
        end
        append!(relevance, relevance_tmp)
        features_mat = hcat(features_mat, features_mat_tmp)
    end

    return queries_count, queries_dict, relevance, features_mat
end



# TODO: keep track of how many iterations are skipped b/c Z is low.

function f_q(theta, x)
    dot(theta, x)
end

function G(s_j)
# p.13 of the paper mentions the following loss which led to extreme values
    2^s_j - 1
# Instead, used the identity as mentioned on p.13
#    s_j
end
#function Z(s)
#    sum(abs(map(G, s)))
#end

function update_ls_surrogate(features_mat, lambda, eta, theta_old, s,
                             x_lines, verbose=false)
# todo: put in an assert that s is in the same order as x_lines. It's done this way when I call things, but good to have a safety.
    G_tmp = map(G, s)
# TODO: modify this
#    Z_tmp = sum(abs(G_tmp))
# TODO: this is same as in ndcg_loss. Reference it somehow?
    pi_alpha_min = get_order(s)
    Z_tmp = sum(G_tmp ./ map((x) -> log(1+x), pi_alpha_min))
    #println("sum with G and Z is ", sum(G_tmp) / Z_tmp)
# check Z_tmp not near zero
    if Z_tmp < 10.0^(-5) || isnan(Z_tmp)
        if verbose
            println("Warning: skipping iteration! Z is low or missing in update_ls_surrogate")
        end
        return theta_old
    end
    tmp_sum = zeros(d)
    x_tmp = Array(Float64, d) # pre-allocated storage
    m = length(x_lines)
    for i in 1:m
# These two lines take up the bulk of time in the program.
# todo: @fastmath? need to update julia... and @inbound?
        x_tmp = features_mat[:, x_lines[i]]
        tmp_sum += x_tmp *
              (f_q(theta_old, x_tmp) - G_tmp[i] / Z_tmp)
    end
    theta = (1/(lambda + 1/eta) * ((1/eta)*theta_old - (1/m) * tmp_sum))
    return theta
end

function normalize_features!(features_mat)
    ncols = size(features_mat)[2]
    for col in 1:ncols
        features_mat[:, col] /= norm(features_mat[:, col])
    end
end


function update_logistic_surrogate(features_mat, lambda, eta, theta_old, x_i, x_j)
# Assuming i≻j
    x_tmp = features_mat[:, x_j] - features_mat[:, x_i]
    theta = 1/(lambda + 1/eta) * (theta_old / eta - 
               x_tmp * logistic(f_q(theta_old, x_tmp)))
    return theta
end


function calc_asympt_scores(queries_dict, relevance)
# Based on Eq.27 in the paper
    scores_dict = Dict()
    for (key in keys(queries_dict))
        line_nums = queries_dict[key]
        m = length(line_nums)
        scores = zeros(m)
        for i in 1:m
            scores[i] = 0
            for j in 1:m
                if j != i
                    x_tmp = relevance[line_nums[i]] - relevance[line_nums[j]]
                    y_tmp = - x_tmp
                    scores[i] += log(1+exp(x_tmp)) - log(1+exp(y_tmp))
                end
            end
        end
        if m > 1
            scores = scores / (m-1)
        end
        @assert !any(isnan(scores)) (key, scores)
        scores_dict[key] = scores
    end
    scores_dict
end


function estimate_scores(relevance, queries_dict, query, k)
# Based on Eq.26 in the paper
    x_lines = queries_dict[query]
    m = length(x_lines)
# TODO: deal with the zero comparisons case! Currently just doing ones()
    comparisons = ones(m,m)
    @assert (m > 1) string("Number of entries for this query is too low! m=", m)
    for i in 1:k
        pair = sample(1:m, 2, replace=false) # slow line(?)
        p_tmp = logistic(relevance[x_lines[pair[1]]] - relevance[x_lines[pair[2]]])
        preference = rand(Bernoulli(p_tmp))
        if preference == 1
            comparisons[pair[1], pair[2]] += 1
        else
            comparisons[pair[2], pair[1]] += 1
        end
    end
    
    scores = zeros(m)
    for i in 1:m
        for j in 1:m
# if i=j, then we're adding 0
            scores[i] += log(comparisons[i,j] / comparisons[j,i])
        end
    end
    if m > 1
        scores = scores / (m-1)
    end
    @assert !any(isnan(scores)) (k, scores)
    return scores
end


function run_updater(lambda, eta_0, k, n_time_steps, queries_dict, relevance,
                     features_mat, which_loss="leastsquares")
    # Initialize parameters
    theta = zeros(d, n_time_steps+1)
    queries = collect(keys(queries_dict))

#=
# TODO: assert that the global variable is held
    if which_loss == "leastsquares_true"
        scores_dict_true = calc_asympt_scores(queries_dict, relevance)
    end
=#

    for t in 1:n_time_steps
# this check takes surprisingly long if apply to all of theta.
        if any(isnan(theta[:,t]))
            println("Quitting at ", t, " because of missing values in theta")
            break
        end

        eta_t = eta_0 / sqrt(t)
        q = sample(queries)
        x_lines = queries_dict[q]
        m = length(x_lines)
        while m < 2
            #println("Warning: m < 2. Skipping.")
            q = sample(queries)
            x_lines = queries_dict[q]
            m = length(x_lines)
        end

        if which_loss == "leastsquares"
# generate the score
            s = estimate_scores(relevance, queries_dict, q, k)

            theta[:,t+1] = update_ls_surrogate(features_mat, lambda, eta_t, theta[:,t],
                       s, x_lines)
        elseif which_loss == "leastsquares_true"
            theta[:,t+1] = update_ls_surrogate(features_mat, lambda, eta_t, theta[:,t],
                       scores_dict_true[q], x_lines)
        elseif which_loss == "surrogate_pair"
            # sampling is similar to estimate_scores()
            pair = sample(1:m, 2, replace=false)
            p_tmp = logistic(relevance[x_lines[pair[1]]] - relevance[x_lines[pair[2]]])
            preference = rand(Bernoulli(p_tmp))
            if preference == 1
# This means pair[1] is preferred to pair[2]
                x_i = x_lines[pair[1]]
                x_j = x_lines[pair[2]]
            else
                x_i = x_lines[pair[2]]
                x_j = x_lines[pair[1]]
            end
            # i ≻ j
            
            theta[:, t+1] = update_logistic_surrogate(features_mat, lambda, eta_t,
                     theta[:,t], x_i, x_j)
        else
            error("which_loss value, ", which_loss, " is unrecognized")
        end
    end

    theta
end

function get_order(alpha::Array)
# return rank vector for array alpha, where high scores come up earlier
    sorted = sortperm(alpha, rev=true)
    m = length(alpha)
# order will invert sorted to get the rank vector
    order = int(zeros(m))
    for i in 1:m
        order[sorted[i]] = i
    end
    return order
end

function ndcg_loss(alpha, s)
# Normalized Discounted Cumulative Game family loss
# p.13, eq.14
    G_s = map((x) -> 2^x - 1, s)
    pi_alpha = get_order(alpha)
    F_pi = map((x) -> log(1+x), pi_alpha)
# Maximizing Z happens when get the order corresponding to s
    pi_alpha_min = get_order(s)
    Z_s = sum(G_s ./ map((x) -> log(1+x), pi_alpha_min)) # todo: check this
    return 1 - (1/Z_s) * sum(G_s ./ F_pi)
end

function regularizer(lambda, theta)
    (lambda/2) * norm(theta)^2
end


# could loop through all queries? doesn't seem too long to do.
function estimate_ndcg_loss(queries, queries_dict, scores_dict_true, features_mat,
                            theta, nsamples=500)
    risk_estimate = 0
    #for i in 1:nsamples
        #q = sample(queries)
    for q in queries
# sample query
#scores_dict_true = calc_asympt_scores(queries_dict, relevance);
        x_lines = queries_dict[q]
        m = length(x_lines)
        if m < 2
            continue
        end
        s = scores_dict_true[q]
        if maximum(s) - minimum(s) < 10.0^(-5) # tolerance
            continue
        end
        alpha = zeros(m)
        for i in 1:m
            alpha[i] = f_q(theta, features_mat[:, x_lines[i]])
        end

        #println(risk_estimate)
        ndcg_loss_tmp = ndcg_loss(alpha, s)
        if isnan(ndcg_loss_tmp)
            println("missing! at query ", q)
            break
        end
        risk_estimate += ndcg_loss(alpha, s)
    end
# average the risk
    #risk_estimate /= nsamples
    risk_estimate /= length(queries)
    return risk_estimate
end

# todo: check query 29716, why are the asymptotic scores zero??

#=
# test:
scores_tmp = scores_dict[15769];
scores_tmp2 = estimate_scores(relevance, queries_dict, 15769, 10^3);
scores_tmp3 = estimate_scores(relevance, queries_dict, 15769, 10^6);
# norm should be small, and smaller with more iterations
norm(scores_tmp - scores_tmp2)
=#


