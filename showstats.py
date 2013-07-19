import sys
import cProfile
import pstats

s = pstats.Stats( sys.argv[1] )
s.sort_stats('cumulative')
s.print_stats()

