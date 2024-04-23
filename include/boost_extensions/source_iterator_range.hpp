/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner   
*/

#pragma once

namespace boost::extensions {

template<typename IteratorType>
struct source_iterator_range {
  private:
    IteratorType _begin, _end;

  public:
    IteratorType begin() { return _begin; }
    IteratorType begin() const { return _begin; }
    IteratorType cbegin() const { return _begin; }

    IteratorType end() { return _end; }
    IteratorType end() const { return _end; }
    IteratorType cend() const { return _end; }

    size_t size() const { return std::distance(cbegin(), cend()); }

    template<class RangeType>
    explicit source_iterator_range(RangeType &r) : _begin(boost::begin(r)), _end(boost::end(r)) {}
};

template<class ForwardRange>
source_iterator_range<decltype(boost::begin(std::declval<ForwardRange &>()))>
make_source_iterator_range(const ForwardRange &r) {
    return source_iterator_range<decltype(boost::begin(std::declval<ForwardRange &>()))>(r);
}

template<class ForwardRange>
source_iterator_range<decltype(boost::begin(std::declval<ForwardRange &>()))>
make_source_iterator_range(ForwardRange &r) {
    return source_iterator_range<decltype(boost::begin(std::declval<ForwardRange &>()))>(r);
}

} // namespace boost::extensions
