from typing import List, Dict, Tuple


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        corpus_list = list(corpus)
        merged_tuples = []

        for _ in range(num_merges):
            frequencies : Dict[Tuple[str, str], int] = {}
            for first, second in zip(corpus_list, corpus_list[1:]):
                frequencies[(first, second)] = 1 + frequencies.get((first, second), 0)
            max_frequency = max(frequencies.values())
            pairs_with_max_frequencies = [pair for pair in frequencies if frequencies[pair] == max_frequency]
            # sort lexicographically in case of equally frequent pairs
            max_freq_pair = sorted(pairs_with_max_frequencies)[0]

            new_corpus_list = []
            idx = 0
            while idx < len(corpus_list) - 1:
                if [corpus_list[idx], corpus_list[idx + 1]] == list(max_freq_pair):
                    new_corpus_list.append("".join(max_freq_pair))
                    idx += 2
                else:
                    new_corpus_list.append(corpus_list[idx])
                    idx += 1
                if idx == len(corpus_list) - 1:
                    new_corpus_list.append(corpus_list[idx])
            merged_tuples.append(list(max_freq_pair))
            corpus_list = new_corpus_list
        return merged_tuples
            


            







