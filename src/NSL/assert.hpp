//
// Created by marcel on 8/20/21.
//

#ifndef NANOSYSTEMLIBRARY_ASSERT_HPP
#define NANOSYSTEMLIBRARY_ASSERT_HPP

#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))

#endif //NANOSYSTEMLIBRARY_ASSERT_HPP