#pragma once
#include <string>

namespace console {

	void init();
	void write(const std::string& text);
	bool read_line(std::string& line);
	int width();

}