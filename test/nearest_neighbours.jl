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

@testset "Nearest-neighbour lookup: a constant column doesn't produce NaN" begin
    # Column 2 is constant across every row -- σ = 0 there. Column 1 makes the ranking
    # unambiguous: row 1 is closest to row 2, farthest from row 4.
    X = [0.0 5.0
         0.1 5.0
         10.0 5.0]

    Xs = SAShE._standardize_columns(X)
    @test !any(isnan, Xs)
    @test Xs[:, 2] == X[:, 2]  # left unscaled, values unchanged

    @test SAShE._nearest_neighbour_indices(X, 1, [1, 2], 1) == [2]
end

@testset "Nearest-neighbour lookup: a NaN column is rejected, naming the right factor" begin
    # std of a NaN column is NaN, and `NaN == 0` is false, so it slips past the
    # constant-column guard and would turn every distance in the sample into NaN.
    # The error must name the factor in the CALLER's numbering, not its position within
    # `coords` -- reporting "column 1" for a NaN in factor 3 defeats the message's purpose.
    X = [0.0 1.0 5.0
         1.0 2.0 NaN
         2.0 3.0 7.0]

    for coords ∈ ([3], [2, 3], [1, 2, 3])
        err = try
            SAShE._nearest_neighbour_indices(X, 1, coords, 1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("column) 3", err.msg)   # factor 3 holds the NaN, whatever `coords` is
    end

    @test SAShE._nearest_neighbour_indices(X, 1, [1, 2], 1) isa Vector{Int64}  # NaN-free subset
end

@testset "Nearest-neighbour lookup: doesn't crash when >12 points share the query's exact position" begin
    # Regression test: with the old fixed k+11 search buffer, a query point could be pushed
    # out of its own "closest points" window by enough exact duplicates, and the code
    # crashed looking for itself in that window. 15 rows identical to the query point (16
    # total) exceeds that old buffer.
    X = vcat(zeros(16, 2), [1.0 1.0])
    l = 1

    @test SAShE._nearest_neighbour_indices(X, l, [1, 2], 1) isa Vector{Int64}
end

@testset "Nearest-neighbour lookup: tie-breaking is uniform beyond the old buffer size" begin
    Random.seed!(97)

    # n_ties duplicate points, all bit-identically tied at the same distance from the
    # query point (row 1, at 0.0) -- more than the old fixed k+11=12 search buffer, so
    # this only passes if every tied point is actually visible. Genuine duplicate rows
    # (not merely "the same multiset of values in a different row order", which an
    # earlier version of this test tried via axis-aligned unit vectors) are needed: even
    # though every column there had the identical *set* of values, `std`'s internal
    # summation isn't order-invariant at the bit level, so the computed per-column σ (and
    # hence the standardized distance) differed by ~1 ULP across columns -- enough for
    # `sortperm` to resolve most "ties" by that noise before the random tiebreak ever
    # mattered, silently defeating the point of this test. Literal duplicates have no
    # such issue: the exact same float, in the exact same column, every time.
    n_ties = 20
    X = reshape(vcat([0.0], fill(1.0, n_ties)), n_ties + 1, 1)

    n_draws = 6000
    counts = Dict(i => 0 for i in 2:(n_ties + 1))
    for _ in 1:n_draws
        winner = only(SAShE._nearest_neighbour_indices(X, 1, [1], 1))
        counts[winner] += 1
    end

    expected = n_draws / n_ties
    @test all(abs(c - expected) < 0.3 * expected for c in values(counts))
end
