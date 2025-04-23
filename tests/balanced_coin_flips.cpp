#define BOOST_TEST_MODULE Balanced_Coin_Flips
#include <boost/test/unit_test.hpp>

#include <bitset>
#include <iostream>
#include "auxiliary/Balanced_Coin_Flips.hpp"

bool thue_morse_gen(long unsigned int n) {
    // std::bitset<sizeof(n)*CHAR_BIT> bits(n);
    unsigned long int bin_sum = 0;
    while (n != 0) {
        bin_sum += n%2;
        n /= 2;
    }
    return bool(bin_sum%2);  // (bits.count()%2);
};

BOOST_AUTO_TEST_CASE(Random_Biased_Coin) {
    std::cout << "True: " << true << " False: " << false << std::endl; 
    Biased_Random Coin;
    std::cout << "Biased Coin: ";
    for (int i = 0 ; i < 200 ; i++) {
        std::cout << Coin.get_flip();
    }
    std::cout << std::endl << std::endl;
};

BOOST_AUTO_TEST_CASE(Thue__Morse) {
    Thue_Morse_Sequence Coin(0);

    std::vector<bool> beginning({0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1});
    std::vector<bool> generated;
    for (long unsigned i = 0 ; i<beginning.size(); i++) {
        bool next = Coin.get_flip();
        generated.emplace_back(next);
        // std::cout << next;
    }
    // std::cout << std::endl;

    BOOST_CHECK( beginning == generated );

    Thue_Morse_Sequence Test_Coin_in_seq(0);
    for (int i = 0 ; i < 200; i++) {
        BOOST_CHECK_EQUAL(Test_Coin_in_seq.get_flip(), thue_morse_gen(i));
        // std::cout << "hi " << i << std::endl;
    }


    for (int i = 0 ; i < 100; i++) {
        unsigned ind = randInt(1048575);
        Thue_Morse_Sequence Test_Coin_random(ind);
        BOOST_CHECK_EQUAL(Test_Coin_random.get_flip(), thue_morse_gen(ind));
        // std::cout << "bye " << i << std::endl;
    }
};


BOOST_AUTO_TEST_CASE(Repeater_Coin) {
    Repeat_Chance Coin;
    std::cout << "Repeater Coin: ";
    for (int i = 0 ; i < 200 ; i++) {
        std::cout << Coin.get_flip();
    }
    std::cout << std::endl << std::endl;
};

BOOST_AUTO_TEST_CASE(Random_Biased_Coin_with_side_bias_1_1) {
    Biased_Random_with_side_bias Coin({1,1});
    int true_count = 0;
    int false_count = 0;
    std::cout << "Biased Coin with side bias 1:1 : ";
    for (int i = 0 ; i < 200 ; i++) {
        bool flip = Coin.get_flip();
        if (flip) {
            true_count++;
        }
        else {
            false_count++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << true_count << " False count: " << false_count << std::endl;
    std::cout << std::endl;
};

BOOST_AUTO_TEST_CASE(Random_Biased_Coin_with_side_bias_1_0) {
    Biased_Random_with_side_bias Coin({1,0});
    int true_count = 0;
    int false_count = 0;
    std::cout << "Biased Coin with side bias 1:0 : ";
    for (int i = 0 ; i < 200 ; i++) {
        bool flip = Coin.get_flip();
        if (flip) {
            true_count++;
        }
        else {
            false_count++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << true_count << " False count: " << false_count << std::endl;
    std::cout << std::endl;
};


BOOST_AUTO_TEST_CASE(Random_Biased_Coin_with_side_bias_0_1) {
    Biased_Random_with_side_bias Coin({0,1});
    int true_count = 0;
    int false_count = 0;
    std::cout << "Biased Coin with side bias 0:1 : ";
    for (int i = 0 ; i < 200 ; i++) {
        bool flip = Coin.get_flip();
        if (flip) {
            true_count++;
        }
        else {
            false_count++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << true_count << " False count: " << false_count << std::endl;
    std::cout << std::endl;
};


BOOST_AUTO_TEST_CASE(Random_Biased_Coin_with_side_bias_3_2) {
    Biased_Random_with_side_bias Coin({3,2});
    int true_count = 0;
    int false_count = 0;
    std::cout << "Biased Coin with side bias 3:2 : ";
    for (int i = 0 ; i < 200 ; i++) {
        bool flip = Coin.get_flip();
        if (flip) {
            true_count++;
        }
        else {
            false_count++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << true_count << " False count: " << false_count << std::endl;
    std::cout << std::endl;
};

BOOST_AUTO_TEST_CASE(Random_Biased_Coin_with_side_bias_3_1) {
    Biased_Random_with_side_bias Coin({3,1});
    int true_count = 0;
    int false_count = 0;
    std::cout << "Biased Coin with side bias 3:1 : ";
    for (int i = 0 ; i < 200 ; i++) {
        bool flip = Coin.get_flip();
        if (flip) {
            true_count++;
        }
        else {
            false_count++;
        }
        std::cout << flip;
    }
    std::cout << std::endl;
    std::cout << "True count: " << true_count << " False count: " << false_count << std::endl;
    std::cout << std::endl;
};