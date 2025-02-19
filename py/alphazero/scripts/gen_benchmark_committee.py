from alphazero.logic.benchmarking import DirectoryOrganizer, BenchmarkCommittee
from alphazero.logic.match_runner import MatchRunner

game = 'c4'
tag = 'benchmark'
n_games = 100
organzier = DirectoryOrganizer(game, tag, db_name='benchmark_i100')
benchmark_committee = BenchmarkCommittee(organzier, load_past_data=True)
matches = MatchRunner.linspace_matches(0, 128, n_iters=100, freq=4, n_games=n_games, \
    model_dir=organzier.model_dir)
benchmark_committee.play_matches(matches)
benchmark_committee.compute_ratings()