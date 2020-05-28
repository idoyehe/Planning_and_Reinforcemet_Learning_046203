import numpy as np

DEALER_BUSTED = 22
PROBABILITY_DICT = \
    {
        2: 1 / 13,
        3: 1 / 13,
        4: 1 / 13,
        5: 1 / 13,
        6: 1 / 13,
        7: 1 / 13,
        8: 1 / 13,
        9: 1 / 13,
        10: 4 / 13,
        11: 1 / 13,
    }


def __calc(first_card, dealer_sum, prob, p):
    if 17 <= dealer_sum <= 21:
        p[first_card][dealer_sum] += prob
        return
    if dealer_sum > 21:
        p[first_card][22] += prob
        return
    for hit_card, hit_probability in PROBABILITY_DICT.items():
        __calc(first_card, dealer_sum + hit_card, prob * hit_probability, p)
    return


def calc_first_card_end_sum_probability():
    first_cards = list(range(2, 12))
    end_sum = list(range(17, 22))
    p = np.zeros(shape=(len(first_cards) + 2, len(end_sum) + 18))
    for first_card in first_cards:
        __calc(first_card, first_card, 1, p)
    return p
