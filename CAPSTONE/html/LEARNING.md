# Bugs I Fixed Today

1. **Dropdown disappearing**
   - Cause: Unknown (maybe Chart.js re-rendering)
   - Fix: Nuclear option - force recreate every second
   - Lesson: Sometimes brute force works

2. **Congestion showing MODERATE at 1 AM**
   - Cause: API returning calculated values instead of closed values
   - Fix: Added early return for closed hours with current: 5
   - Lesson: Check operating hours BEFORE calculations

3. **Main station name not updating**
   - Cause: Typo - getElementsById instead of getElementById
   - Fix: Corrected method name
   - Lesson: One character can break everything

4. **recreate update best time**
    