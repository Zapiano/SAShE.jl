@testset "Nearest-neighbour lookup: correctness" begin
    # Points laid out on a line: X[i] = i, so neighbour ranking is unambiguous.
    X = reshape(Float64.(1:10), 10, 1)

    @test SAShE._nearest_neighbour_indices(X, 1, [1], 1) == [2]
    @test Set(SAShE._nearest_neighbour_indices(X, 5, [1], 2)) == Set([4, 6])
    @test Set(SAShE._nearest_neighbour_indices(X, 1, [1], 3)) == Set([2, 3, 4])
    @test Set(SAShE._nearest_neighbour_indices(X, 10, [1], 3)) == Set([7, 8, 9])
end

@testset "Nearest-neighbour lookup: coordinate-subset restriction" begin
    # Column 1 makes rows 1 and 2 close; column 2 makes rows 1 and 3 close.
    # Restricting to different coordinate subsets should change the answer.
    X = [0.0 0.0
         0.1 100.0
         100.0 0.1]

    @test SAShE._nearest_neighbour_indices(X, 1, [1], 1) == [2]
    @test SAShE._nearest_neighbour_indices(X, 1, [2], 1) == [3]
end

@testset "Nearest-neighbour lookup: standardization prevents scale domination" begin
    Random.seed!(13)

    X = [0.0 0.0
         1.0 0.0
         0.0 5.0
         0.0 1000.0]

    # Independent reference: standardize by hand (not reusing the code under test) and
    # find the true nearest neighbour of row 1 among rows 2:4 under standardized distance.
    σ = [std(X[:, j]) for j in 1:2]
    Xs = X ./ reshape(σ, 1, :)
    std_dists = [sqrt(sum((Xs[1, :] .- Xs[i, :]) .^ 2)) for i in 2:4]
    expected = argmin(std_dists) + 1

    # Under raw (unstandardized) Euclidean distance, the answer would differ — this
    # confirms the scenario actually exercises standardization, not a coincidence.
    raw_dists = [sqrt(sum((X[1, :] .- X[i, :]) .^ 2)) for i in 2:4]
    @test argmin(raw_dists) + 1 != expected

    @test only(SAShE._nearest_neighbour_indices(X, 1, [1, 2], 1)) == expected
end

@testset "Nearest-neighbour lookup: tie-breaking is uniform" begin
    Random.seed!(2468)

    # Four points equidistant from the query point (row 1).
    X = [0.0 0.0
         1.0 0.0
         -1.0 0.0
         0.0 1.0
         0.0 -1.0]

    n_draws = 4000
    counts = Dict(i => 0 for i in 2:5)
    for _ in 1:n_draws
        winner = only(SAShE._nearest_neighbour_indices(X, 1, [1, 2], 1))
        counts[winner] += 1
    end

    expected = n_draws / length(counts)
    @test all(abs(c - expected) < 0.15 * expected for c in values(counts))
end

@testset "Nearest-neighbour lookup: error on N_I > N - 1" begin
    X = reshape(Float64.(1:5), 5, 1)  # 5 points ⇒ 4 other points available

    @test_throws ArgumentError SAShE._nearest_neighbour_indices(X, 1, [1], 5)
    @test Set(SAShE._nearest_neighbour_indices(X, 1, [1], 4)) == Set([2, 3, 4, 5])  # k == N - 1 still works
end
